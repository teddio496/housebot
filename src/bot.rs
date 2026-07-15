//! Discord interface (serenity): message routing, `!`-commands, streaming progress
//! updates, secret redaction, and code file uploads.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use regex::Regex;
use serenity::all::{
    ButtonStyle, Command, CommandDataOptionValue, CommandOptionType, ComponentInteractionDataKind,
    Context, CreateActionRow, CreateAllowedMentions, CreateAttachment, CreateButton, CreateCommand,
    CreateCommandOption, CreateEmbed, CreateInteractionResponse, CreateInteractionResponseMessage,
    CreateSelectMenu, CreateSelectMenuKind, CreateSelectMenuOption, EditInteractionResponse,
    EditMessage, EventHandler, GatewayIntents, Interaction, Message, Ready, UserId,
};
use serenity::builder::CreateMessage;
use serenity::Client;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::agent::{
    Agent, AgentControlAction, AgentHooks, AgentRequest, AgentResult, MediaData, NoHooks,
};
use crate::bot_config::{ServerConfigStore, UserConfigStore};
pub use crate::bot_response::SecretRedactor;
use crate::channel_log::ChannelLog;
use crate::coding_agent::catalog::{AgentCatalog, CodingAgent};
use crate::coding_agent::issue::{build_issue_body, dispatch_labels};
use crate::coding_agent::pending::{DiscordMessageRef, DispatchStage, PendingJobStore};
use crate::config;
use crate::discord_bridge::DiscordBridge;
use crate::history::History;
use crate::llm::ThinkingMode;
use crate::lua_engine;
use crate::memory::Memory;
use crate::message_log::MessageLog;
use crate::notes::Notes;
use crate::profile::ProfileStore;
use crate::rate_limit::RateLimiter;
use crate::skills::Skills;

pub use crate::bot_commands::{erase_data_command, note_command, skill_command, stats_command};
use crate::bot_formatting::append_tool_summary;
pub use crate::bot_formatting::{extract_code_files, lang_ext, split_text, tool_hint};

const MAX_MESSAGE_LENGTH: usize = 2000;
const EMBED_DESCRIPTION_LIMIT: usize = 4096;
const PAGINATION_PREFIX: &str = "housebot_labs_page:";
const DEVELOP_PREFIX: &str = "develop:";

struct PaginatedResponse {
    owner_id: u64,
    pages: Vec<String>,
}

fn compact_progress(stage: usize, detail: Option<&str>) -> String {
    let filled = (stage / 10).min(10);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled));
    match detail {
        Some(detail) => format!("🧠 **Compacting conversation**\n`[{bar}] {stage}%` — {detail}"),
        None => format!("🧠 **Compacting conversation**\n`[{bar}] {stage}%`"),
    }
}

enum CompactProgressTarget {
    Message {
        ctx: Context,
        channel_id: serenity::all::ChannelId,
        message_id: serenity::all::MessageId,
    },
    Interaction {
        ctx: Context,
        command: Box<serenity::all::CommandInteraction>,
    },
}

struct CompactProgressHooks(CompactProgressTarget);

#[async_trait]
impl AgentHooks for CompactProgressHooks {
    async fn on_progress(&self, line: &str) {
        let Some(rest) = line.strip_prefix("compact:") else {
            return;
        };
        let (stage, detail) = rest.split_once(':').unwrap_or((rest, ""));
        let Ok(stage) = stage.parse::<usize>() else {
            return;
        };
        let content = compact_progress(stage, (!detail.is_empty()).then_some(detail));
        match &self.0 {
            CompactProgressTarget::Message {
                ctx,
                channel_id,
                message_id,
            } => {
                let _ = channel_id
                    .edit_message(&ctx.http, *message_id, EditMessage::new().content(content))
                    .await;
            }
            CompactProgressTarget::Interaction { ctx, command } => {
                let _ = command
                    .edit_response(&ctx.http, EditInteractionResponse::new().content(content))
                    .await;
            }
        }
    }
}

struct ResponseProgressHooks {
    ctx: Context,
    channel_id: serenity::all::ChannelId,
    message_id: serenity::all::MessageId,
    generating: AtomicBool,
}

impl ResponseProgressHooks {
    fn new(ctx: &Context, progress: &Message) -> Self {
        Self {
            ctx: ctx.clone(),
            channel_id: progress.channel_id,
            message_id: progress.id,
            generating: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl AgentHooks for ResponseProgressHooks {
    async fn on_text_stream(&self, _partial: &str) {
        if self.generating.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self
            .channel_id
            .edit_message(
                &self.ctx.http,
                self.message_id,
                EditMessage::new().content("⚙️ **Generating...**"),
            )
            .await;
    }
}
// ── pure helpers ─────────────────────────────────────────────────────────────

static URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://[^\s<>]+|www\.[^\s<>]+").unwrap());

/// Tracks which (channel, user) conversations are still within the idle window.
pub struct ConversationTracker {
    default_idle_timeout: Duration,
    last_active: std::collections::HashMap<(u64, u64), (Instant, Duration)>,
}

impl ConversationTracker {
    pub fn new(idle_timeout: Duration) -> Self {
        Self {
            default_idle_timeout: idle_timeout,
            last_active: std::collections::HashMap::new(),
        }
    }

    pub fn is_active(&self, channel_id: u64, user_id: u64, now: Instant) -> bool {
        match self.last_active.get(&(channel_id, user_id)) {
            Some(&(t, timeout)) => now.duration_since(t) <= timeout,
            None => false,
        }
    }

    /// Remove an expired entry; return whether one existed.
    pub fn pop_timed_out(&mut self, channel_id: u64, user_id: u64, now: Instant) -> bool {
        let key = (channel_id, user_id);
        if let Some(&(t, timeout)) = self.last_active.get(&key) {
            if now.duration_since(t) > timeout {
                self.last_active.remove(&key);
                return true;
            }
        }
        false
    }

    pub fn mark_active(&mut self, channel_id: u64, user_id: u64, now: Instant, timeout: Duration) {
        self.last_active
            .insert((channel_id, user_id), (now, timeout));
    }

    pub fn remove(&mut self, channel_id: u64, user_id: u64) {
        self.last_active.remove(&(channel_id, user_id));
    }

    pub fn default_timeout(&self) -> Duration {
        self.default_idle_timeout
    }
}

// ── serenity handler ─────────────────────────────────────────────────────────

/// The Discord client state.
pub struct HouseBot {
    agent: Arc<Agent>,
    redactor: Arc<SecretRedactor>,
    notes: Notes,
    skills: Skills,
    memory: Memory,
    history: History,
    profile_store: ProfileStore,
    message_log: MessageLog,
    server_cfg: ServerConfigStore,
    user_cfg: UserConfigStore,
    conversations: Mutex<ConversationTracker>,
    processing: Mutex<HashSet<u64>>,
    responded: Mutex<VecDeque<u64>>,
    proactive_cooldowns: Mutex<HashMap<(u64, u64), Instant>>,
    paginated: Mutex<HashMap<String, PaginatedResponse>>,
    reminder_started: AtomicBool,
    chat_rate_limiter: RateLimiter,
    lua_rate_limiter: RateLimiter,
    /// Shared with `Agent` — holds pending coding-agent dispatch jobs.
    pending_jobs: Arc<PendingJobStore>,
    /// Catalog of agents, models, and effort levels.
    catalog: AgentCatalog,
    /// Shared with `Agent` — provides Discord API access to the agent tools.
    discord: Arc<DiscordBridge>,
    /// Logs all guild channel messages for the search_messages tool.
    channel_log: ChannelLog,
}

impl HouseBot {
    /// Build the bot from environment configuration.
    pub fn new(agent: Arc<Agent>, discord: Arc<DiscordBridge>) -> Self {
        let idle = Duration::from_secs(config::env_parse("CONVERSATION_IDLE_TIMEOUT", 300));
        let chat_rate_max: usize = config::env_parse("CHAT_RATE_LIMIT_MAX", 20);
        let chat_rate_window =
            Duration::from_secs(config::env_parse("CHAT_RATE_LIMIT_WINDOW_SECS", 60u64));
        let pending_jobs = agent.pending_jobs();
        let memory = agent.memory();
        Self {
            agent,
            redactor: Arc::new(SecretRedactor::from_env()),
            notes: Notes::default(),
            skills: Skills::default(),
            memory,
            history: History::default(),
            profile_store: ProfileStore::default(),
            message_log: MessageLog::default(),
            server_cfg: ServerConfigStore::default(),
            user_cfg: UserConfigStore::default(),
            conversations: Mutex::new(ConversationTracker::new(idle)),
            processing: Mutex::new(HashSet::new()),
            responded: Mutex::new(VecDeque::with_capacity(200)),
            proactive_cooldowns: Mutex::new(HashMap::new()),
            paginated: Mutex::new(HashMap::new()),
            reminder_started: AtomicBool::new(false),
            chat_rate_limiter: RateLimiter::new(chat_rate_max, chat_rate_window),
            lua_rate_limiter: RateLimiter::new(
                config::env_parse("LUA_RATE_LIMIT_MAX", 6),
                Duration::from_secs(config::env_parse("LUA_RATE_LIMIT_WINDOW_SECS", 60u64)),
            ),
            pending_jobs,
            catalog: AgentCatalog::load_embedded(),
            discord,
            channel_log: ChannelLog::default(),
        }
    }

    async fn already_seen(&self, id: u64) -> bool {
        let mut processing = self.processing.lock().await;
        let responded = self.responded.lock().await;
        if processing.contains(&id) || responded.contains(&id) {
            return true;
        }
        processing.insert(id);
        false
    }

    async fn mark_done(&self, id: u64) {
        self.processing.lock().await.remove(&id);
        let mut responded = self.responded.lock().await;
        if responded.len() >= 200 {
            responded.pop_front();
        }
        responded.push_back(id);
    }

    /// Handle `/new`, `/reset`, `!new`, and `!reset` — they are all aliases.
    async fn handle_new(&self, channel_id: u64, user_id: u64) -> String {
        tracing::info!(target: "housebot::commands", user_id, "Session reset requested");
        self.agent.reset_session(&user_id.to_string()).await;
        self.conversations.lock().await.remove(channel_id, user_id);
        let name = self
            .profile_store
            .load(user_id)
            .await
            .best_name()
            .to_string();
        format!("New conversation started, {name}. Your previous conversation history has been cleared.")
    }

    async fn proactive_cooldown_allows(&self, channel_id: u64, user_id: u64) -> bool {
        let now = Instant::now();
        let cooldown = Duration::from_secs(config::env_parse(
            "PROACTIVE_ASSISTANCE_COOLDOWN_SECS",
            300u64,
        ));
        let mut cooldowns = self.proactive_cooldowns.lock().await;
        if cooldowns
            .get(&(channel_id, user_id))
            .is_some_and(|last| now.duration_since(*last) < cooldown)
        {
            return false;
        }
        cooldowns.insert((channel_id, user_id), now);
        true
    }

    async fn respond(&self, ctx: &Context, msg: &Message, content: &str) {
        let _ = reply_no_ping(ctx, msg, content).await;
    }
}

/// Handle a `/config` interaction, returning the reply text (sent ephemerally).
async fn handle_config_interaction(
    server_cfg: &ServerConfigStore,
    user_cfg: &UserConfigStore,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
    guild_id: Option<u64>,
) -> String {
    let Some(top) = options.first() else {
        return "No subcommand provided.".into();
    };

    match top.name.as_str() {
        "channel" => {
            let Some(gid) = guild_id else {
                return "Channel configuration is only available in servers, not DMs.".into();
            };
            let sub_opts = match &top.value {
                CommandDataOptionValue::SubCommandGroup(opts) => opts,
                _ => return "Unexpected option structure.".into(),
            };
            let Some(sub) = sub_opts.first() else {
                return "No channel subcommand provided.".into();
            };
            match sub.name.as_str() {
                "list" => {
                    let cfg = server_cfg.load(gid).await;
                    if cfg.allowed_channel_ids.is_empty() {
                        "I'm allowed to respond in **all channels** (no restriction set). Follow-up replies are disabled until you add explicit reply channels.".into()
                    } else {
                        let ids: Vec<String> = cfg
                            .allowed_channel_ids
                            .iter()
                            .map(|id| format!("<#{id}>"))
                            .collect();
                        format!("Allowed channels: {}", ids.join(", "))
                    }
                }
                "clear" => {
                    let mut cfg = server_cfg.load(gid).await;
                    cfg.allowed_channel_ids.clear();
                    if server_cfg.save(gid, &cfg).await.is_err() {
                        return "Error: failed to save config.".into();
                    }
                    "✅ Channel restriction cleared — I'll respond in all channels, but follow-up replies are disabled until you add explicit reply channels.".into()
                }
                action @ ("add" | "remove") => {
                    let channel_opts = match &sub.value {
                        CommandDataOptionValue::SubCommand(opts) => opts,
                        _ => return "Unexpected option structure.".into(),
                    };
                    let channel_id =
                        channel_opts
                            .iter()
                            .find(|o| o.name == "channel")
                            .and_then(|o| match &o.value {
                                CommandDataOptionValue::Channel(c) => Some(c.get()),
                                _ => None,
                            });
                    let Some(cid) = channel_id else {
                        return "Please provide a valid channel.".into();
                    };
                    let mut cfg = server_cfg.load(gid).await;
                    if action == "add" {
                        cfg.allowed_channel_ids.insert(cid);
                        if server_cfg.save(gid, &cfg).await.is_err() {
                            return "Error: failed to save config.".into();
                        }
                        format!("✅ <#{cid}> added to the allowlist.")
                    } else {
                        let removed = cfg.allowed_channel_ids.remove(&cid);
                        if server_cfg.save(gid, &cfg).await.is_err() {
                            return "Error: failed to save config.".into();
                        }
                        if removed {
                            format!("✅ <#{cid}> removed from the allowlist.")
                        } else {
                            format!("<#{cid}> was not in the allowlist.")
                        }
                    }
                }
                other => format!("Unknown channel subcommand `{other}`."),
            }
        }

        "personality" => {
            let sub_opts = match &top.value {
                CommandDataOptionValue::SubCommand(opts) => opts,
                _ => return "Unexpected option structure.".into(),
            };
            let text = sub_opts
                .iter()
                .find(|o| o.name == "text")
                .and_then(|o| match &o.value {
                    CommandDataOptionValue::String(s) => Some(s.clone()),
                    _ => None,
                });
            let mut cfg = user_cfg.load(author_id).await;
            match text {
                None => {
                    cfg.personality = None;
                    if user_cfg.save(author_id, &cfg).await.is_err() {
                        return "Error: failed to save config.".into();
                    }
                    "✅ Personality cleared — I'll use my default behaviour.".into()
                }
                Some(ref s) if s.trim().is_empty() => {
                    cfg.personality = None;
                    if user_cfg.save(author_id, &cfg).await.is_err() {
                        return "Error: failed to save config.".into();
                    }
                    "✅ Personality cleared — I'll use my default behaviour.".into()
                }
                Some(s) => {
                    cfg.personality = Some(s.clone());
                    if user_cfg.save(author_id, &cfg).await.is_err() {
                        return "Error: failed to save config.".into();
                    }
                    format!("✅ Personality set:\n> {}", s.replace('\n', "\n> "))
                }
            }
        }

        "followup" => {
            let sub_opts = match &top.value {
                CommandDataOptionValue::SubCommand(opts) => opts,
                _ => return "Unexpected option structure.".into(),
            };
            let enabled =
                sub_opts
                    .iter()
                    .find(|o| o.name == "enabled")
                    .and_then(|o| match &o.value {
                        CommandDataOptionValue::Boolean(b) => Some(*b),
                        _ => None,
                    });
            let timeout =
                sub_opts
                    .iter()
                    .find(|o| o.name == "timeout")
                    .and_then(|o| match &o.value {
                        CommandDataOptionValue::Integer(n) => Some(*n),
                        _ => None,
                    });
            let Some(enabled) = enabled else {
                return "Please specify `enabled`.".into();
            };
            let mut cfg = user_cfg.load(author_id).await;
            cfg.followup_enabled = enabled;
            if let Some(secs) = timeout {
                if secs < 1 {
                    return "Timeout must be at least 1 second.".into();
                }
                cfg.followup_timeout_secs = secs as u64;
            }
            if user_cfg.save(author_id, &cfg).await.is_err() {
                return "Error: failed to save config.".into();
            }
            let status = if enabled { "enabled" } else { "disabled" };
            format!(
                "✅ Follow-up replies {status} (timeout: {}s).",
                cfg.followup_timeout_secs
            )
        }

        other => format!("Unknown config option `{other}`."),
    }
}

/// Handle an `/effort` interaction: show or change the user's thinking mode.
async fn handle_effort_interaction(
    user_cfg: &UserConfigStore,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
) -> String {
    let level = options
        .iter()
        .find(|o| o.name == "level")
        .and_then(|o| match &o.value {
            CommandDataOptionValue::String(s) => Some(s.clone()),
            _ => None,
        });
    let mut cfg = user_cfg.load(author_id).await;
    let Some(level) = level else {
        let lines: Vec<String> = ThinkingMode::ALL
            .into_iter()
            .map(|mode| {
                let marker = if mode == cfg.thinking_mode {
                    " ←"
                } else {
                    ""
                };
                format!("• **{mode}** — {}{marker}", mode.budget_label())
            })
            .collect();
        return format!(
            "**Thinking effort:** currently **{}** ({}).\n{}\nUse `/effort level:<mode>` to change it.",
            cfg.thinking_mode,
            cfg.thinking_mode.budget_label(),
            lines.join("\n")
        );
    };
    let Ok(mode) = level.parse::<ThinkingMode>() else {
        return format!("Unknown effort level `{level}`. Options: low, medium, high, xhigh, max.");
    };
    cfg.thinking_mode = mode;
    if let Err(error) = user_cfg.save(author_id, &cfg).await {
        tracing::error!(target: "housebot::commands", user_id = author_id, %error, "Failed to save effort setting");
        return "Error: failed to save config.".into();
    }
    tracing::info!(target: "housebot::commands", user_id = author_id, mode = %mode, "Thinking effort updated");
    format!(
        "✅ Thinking effort set to **{mode}** ({}).",
        mode.budget_label()
    )
}

/// Handle a `/status` interaction: show the user's current settings at a glance.
async fn handle_status_interaction(user_cfg: &UserConfigStore, author_id: u64) -> String {
    let cfg = user_cfg.load(author_id).await;
    let effort = format!(
        "**{}** — {}",
        cfg.thinking_mode,
        cfg.thinking_mode.budget_label()
    );
    let followup = if cfg.followup_enabled {
        format!("enabled (timeout: {}s)", cfg.followup_timeout_secs)
    } else {
        "disabled".to_string()
    };
    let personality = match &cfg.personality {
        Some(p) if !p.trim().is_empty() => format!("> {}", p.trim().replace('\n', "\n> ")),
        _ => "default".to_string(),
    };
    format!(
        "**Your current settings:**\n• Effort level: {effort}\n• Follow-up replies: {followup}\n• Personality: {personality}\n\nUse `/effort` to change the thinking effort level."
    )
}

async fn handle_labs_interaction(
    user_cfg: &UserConfigStore,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
) -> String {
    let mut cfg = user_cfg.load(author_id).await;
    let Some(top) = options.first() else {
        return "Choose a labs feature. Use `/labs list` to see available features.".into();
    };
    match top.name.as_str() {
        "list" => format!(
            "**Labs features**\n• Pagination: {}",
            if cfg.labs_pagination_enabled {
                "enabled"
            } else {
                "disabled"
            }
        ),
        "pagination" => {
            let CommandDataOptionValue::SubCommand(sub_opts) = &top.value else {
                return "Unexpected option structure.".into();
            };
            let Some(enabled) =
                sub_opts
                    .iter()
                    .find(|o| o.name == "enabled")
                    .and_then(|o| match &o.value {
                        CommandDataOptionValue::Boolean(value) => Some(*value),
                        _ => None,
                    })
            else {
                return "Please specify `enabled`.".into();
            };
            cfg.labs_pagination_enabled = enabled;
            if let Err(error) = user_cfg.save(author_id, &cfg).await {
                tracing::error!(target: "housebot::labs::pagination", user_id = author_id, %error, "Failed to save pagination setting");
                return "Error: failed to save labs configuration.".into();
            }
            tracing::info!(target: "housebot::labs::pagination", user_id = author_id, enabled, "Updated pagination setting");
            format!(
                "✅ Paginated responses {}.",
                if enabled { "enabled" } else { "disabled" }
            )
        }
        other => format!("Unknown labs feature `{other}`. Use `/labs list`."),
    }
}

/// Handle a `/profile` interaction: show or clear profile data.
async fn handle_profile_interaction(
    profile_store: &ProfileStore,
    memory: &Memory,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
    guild_id: Option<u64>,
) -> String {
    let profile = profile_store.load(author_id).await;
    let subcommand = options.first().map(|o| o.name.as_str());
    match subcommand {
        Some("clear") => {
            let mut profile = profile_store.load(author_id).await;
            profile.clear_learned();
            let profile_result = profile_store.save(author_id, &profile).await;
            let memory_result = memory.clear(author_id.to_string()).await;
            if profile_result.is_err() || memory_result.is_err() {
                "⚠️ Could not clear all learned profile data.".into()
            } else {
                "✅ Profile learned data and memory cleared. Your Discord identity is preserved."
                    .into()
            }
        }
        _ => {
            let name = profile.best_name();
            let tags: Vec<String> = profile
                .tags
                .iter()
                .map(|t| t.as_str().to_string())
                .collect();
            let actions = profile.quick_actions();
            let mut lines = vec![
                format!("**Profile for {name}**"),
                format!("Username: {}", profile.username),
                format!("Display name: {}", profile.display_name),
                format!(
                    "Guild: {}",
                    guild_id
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| "DM".to_string())
                ),
            ];
            if !profile.nickname.is_empty() {
                lines.push(format!("Nickname: {}", profile.nickname));
            }
            if !profile.avatar_url.is_empty() {
                lines.push("Avatar: (set)".to_string());
            }
            if !tags.is_empty() {
                lines.push(format!("Tags: {}", tags.join(", ")));
            }
            if !actions.is_empty() {
                let action_strs: Vec<String> =
                    actions.iter().map(|(k, v)| format!("{k}: {v}")).collect();
                lines.push(format!("Quick actions: {}", action_strs.join(", ")));
            }
            lines.join("\n")
        }
    }
}

/// Handle a `/history` interaction: show or clear history.
async fn handle_history_interaction(
    history: &History,
    profile_store: &ProfileStore,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
    _guild_id: Option<u64>,
) -> String {
    let profile = profile_store.load(author_id).await;
    let name = profile.best_name();
    let subcommand = options.first().map(|o| o.name.as_str());
    match subcommand {
        Some("clear") => {
            let _ = history.clear(author_id.to_string()).await;
            format!("✅ Conversation history cleared for {name}.")
        }
        _ => {
            let hist = history.load(author_id.to_string()).await;
            render_history(&profile, &hist)
        }
    }
}

fn render_history(profile: &crate::profile::UserProfile, hist: &[serde_json::Value]) -> String {
    let name = profile.best_name();
    let mut lines = vec![
        format!("**History for {name}**"),
        "Scope: all servers and channels where you used housebot".to_string(),
    ];

    let profile_bits: Vec<String> = profile
        .tags
        .iter()
        .map(|tag| tag.as_str().to_string())
        .collect();
    if !profile_bits.is_empty() {
        lines.push(format!("Profile interests: {}", profile_bits.join(", ")));
    }

    if hist.is_empty() {
        lines.push("No conversation history yet.".to_string());
        return lines.join("\n");
    }

    let turn_count = hist
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .count();
    let mut recent: Vec<&serde_json::Value> = hist
        .iter()
        .rev()
        .filter(|m| m.get("content").and_then(|c| c.as_str()).is_some())
        .take(10)
        .collect();
    recent.reverse();

    lines.push(format!(
        "Total messages: {} ({} turns)",
        hist.len(),
        turn_count
    ));
    lines.push("Recent interactions:".to_string());
    for msg in recent {
        let role = msg["role"].as_str().unwrap_or("?");
        let content = msg["content"].as_str().unwrap_or("");
        let preview: String = content.chars().take(80).collect();
        let location = msg
            .get("discord_context")
            .and_then(|ctx| ctx.get("channel_id"))
            .and_then(|id| id.as_u64())
            .map(|id| format!(" in <#{id}>"))
            .unwrap_or_default();
        let timestamp = msg
            .get("discord_context")
            .and_then(|ctx| ctx.get("timestamp"))
            .and_then(|value| value.as_str())
            .and_then(|value| value.get(..10))
            .map(|date| format!(" on {date}"))
            .unwrap_or_default();
        lines.push(format!("[{role}{location}{timestamp}] {preview}"));
    }
    if hist.len() > 10 {
        lines.push(format!("... and {} more messages", hist.len() - 10));
    }
    lines.join("\n")
}

/// Handle a `/privacy` interaction: view or change privacy settings.
async fn handle_privacy_interaction(
    user_cfg: &UserConfigStore,
    memory: &Memory,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
) -> String {
    let subcommand = options.first().map(|o| o.name.as_str());
    match subcommand {
        None | Some("status") => {
            let cfg = user_cfg.load(author_id).await;
            let mem_content = memory.load(author_id.to_string()).await;
            let deep_memory = if cfg.deep_memory_enabled {
                if mem_content.trim().is_empty() {
                    "enabled (no memories stored yet)".to_string()
                } else {
                    format!(
                        "enabled ({} bytes stored — use `/memory show` to view)",
                        mem_content.len()
                    )
                }
            } else {
                "disabled".to_string()
            };
            let proactive = if cfg.proactive_assistance_enabled {
                "enabled"
            } else {
                "disabled"
            };
            format!(
                "**Privacy settings:**\n• Deep memory: {deep_memory} (persistent facts across sessions)\n• Proactive assistance: {proactive} (bot may respond without ping)\n\nUse `/privacy deep_memory enabled:true` or `/privacy proactive enabled:false` to change."
            )
        }
        Some("deep_memory") => {
            let sub_opts = match &options[0].value {
                serenity::all::CommandDataOptionValue::SubCommand(opts) => opts,
                _ => return "Unexpected option structure.".into(),
            };
            let enabled =
                sub_opts
                    .iter()
                    .find(|o| o.name == "enabled")
                    .and_then(|o| match &o.value {
                        serenity::all::CommandDataOptionValue::Boolean(b) => Some(*b),
                        _ => None,
                    });
            let Some(enabled) = enabled else {
                return "Please specify `enabled`.".into();
            };
            let mut cfg = user_cfg.load(author_id).await;
            cfg.deep_memory_enabled = enabled;
            if user_cfg.save(author_id, &cfg).await.is_err() {
                return "Error: failed to save config.".into();
            }
            if enabled {
                "✅ Deep memory enabled. I will now remember important facts about you across conversations. Use `/memory show` to see what I currently remember.".into()
            } else {
                "✅ Deep memory disabled. I will no longer save facts between sessions (your current memories are kept but won't be updated).".into()
            }
        }
        Some("proactive") => {
            let sub_opts = match &options[0].value {
                serenity::all::CommandDataOptionValue::SubCommand(opts) => opts,
                _ => return "Unexpected option structure.".into(),
            };
            let enabled =
                sub_opts
                    .iter()
                    .find(|o| o.name == "enabled")
                    .and_then(|o| match &o.value {
                        serenity::all::CommandDataOptionValue::Boolean(b) => Some(*b),
                        _ => None,
                    });
            let Some(enabled) = enabled else {
                return "Please specify `enabled`.".into();
            };
            let mut cfg = user_cfg.load(author_id).await;
            cfg.proactive_assistance_enabled = enabled;
            if user_cfg.save(author_id, &cfg).await.is_err() {
                return "Error: failed to save config.".into();
            }
            format!(
                "✅ Proactive assistance {}.",
                if enabled { "enabled" } else { "disabled" }
            )
        }
        other => {
            format!("Unknown privacy option `{other:?}`. Use `/privacy` to see available options.")
        }
    }
}

/// Handle a `/memory` interaction: view or clear the bot's memory about the user.
async fn handle_memory_interaction(
    memory: &Memory,
    options: &[serenity::all::CommandDataOption],
    author_id: u64,
) -> String {
    let subcommand = options.first().map(|o| o.name.as_str());
    match subcommand {
        None | Some("show") => {
            let content = memory.load(author_id.to_string()).await;
            if content.trim().is_empty() {
                "No memories stored yet. Enable deep memory with `/privacy deep_memory enabled:true` and I will start remembering things about you across conversations.".into()
            } else {
                format!("**What I remember about you:**\n{content}")
            }
        }
        Some("clear") => match memory.clear(author_id.to_string()).await {
            Ok(()) => "✅ Your memory has been cleared. I no longer remember anything about you from past sessions.".into(),
            Err(_) => "⚠️ Failed to clear memory. Please try again.".into(),
        },
        other => format!("Unknown memory subcommand `{other:?}`. Use `/memory show` or `/memory clear`."),
    }
}

async fn reply_no_ping(ctx: &Context, msg: &Message, content: &str) -> serenity::Result<Message> {
    let builder = CreateMessage::new()
        .content(content)
        .reference_message(msg)
        .allowed_mentions(CreateAllowedMentions::new());
    msg.channel_id.send_message(&ctx.http, builder).await
}

fn help_response() -> String {
    crate::tools::features::features_text().to_string()
}

fn is_proactive_candidate(content: &str) -> bool {
    let normalized = content.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }
    normalized.contains('?')
        || normalized.starts_with("how ")
        || normalized.starts_with("what ")
        || normalized.starts_with("where ")
        || normalized.starts_with("when ")
        || normalized.starts_with("why ")
        || normalized.starts_with("can you ")
        || normalized.starts_with("could you ")
        || normalized.starts_with("remind me ")
        || normalized.starts_with("how do i ")
        || normalized.starts_with("what can you do")
}

fn compact_done_message(deep_memory_enabled: bool) -> &'static str {
    if deep_memory_enabled {
        "✅ Conversation compacted into memory. A new session has started."
    } else {
        "✅ Conversation cleared without saving a memory summary. A new session has started."
    }
}

fn commit_hash_response(sha: Option<&str>) -> String {
    match sha.filter(|sha| !sha.is_empty()) {
        Some(sha) => format!("Running commit: `{sha}`"),
        None => "Running commit is unavailable for this build.".into(),
    }
}

/// Wrap `/lua` output in a code fence sized to fit a single Discord message.
fn format_lua_reply(output: &str) -> String {
    let sanitized = output.replace("```", "`\u{200b}``");
    let budget = MAX_MESSAGE_LENGTH - "```\n\n```".chars().count();
    let body: String = if sanitized.chars().count() > budget {
        let mut truncated: String = sanitized.chars().take(budget - 1).collect();
        truncated.push('…');
        truncated
    } else {
        sanitized
    };
    format!("```\n{body}\n```")
}

async fn respond_ephemeral(ctx: &Context, cmd: &serenity::all::CommandInteraction, content: &str) {
    let response = CreateInteractionResponse::Message(
        CreateInteractionResponseMessage::new()
            .content(content)
            .ephemeral(true),
    );
    if let Err(e) = cmd.create_response(&ctx.http, response).await {
        tracing::warn!("Failed to send interaction response: {e}");
    }
}

#[serenity::async_trait]
impl EventHandler for HouseBot {
    async fn ready(&self, ctx: Context, ready: Ready) {
        tracing::info!("Logged in as {} (ID: {})", ready.user.name, ready.user.id);
        self.discord.set_http(ctx.http.clone()).await;

        // Register the /config global slash command.
        let config_cmd = CreateCommand::new("config")
            .description("Configure bot settings")
            // ── channel subcommand group ─────────────────────────────────────
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommandGroup,
                    "channel",
                    "Manage which channels the bot responds in (server-wide)",
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "list",
                    "Show the current channel allowlist",
                ))
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "add",
                        "Add a channel to the allowlist",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::Channel,
                            "channel",
                            "The channel to allow",
                        )
                        .required(true),
                    ),
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "remove",
                        "Remove a channel from the allowlist",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::Channel,
                            "channel",
                            "The channel to remove",
                        )
                        .required(true),
                    ),
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "clear",
                    "Remove all channel restrictions (bot responds everywhere)",
                )),
            )
            // ── personality subcommand ───────────────────────────────────────
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "personality",
                    "Set or clear your personal bot personality / tone override",
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::String,
                    "text",
                    "Personality description (omit to clear your override)",
                )),
            )
            // ── followup subcommand ──────────────────────────────────────────
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "followup",
                    "Control whether the bot replies without a ping during active conversations",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::Boolean,
                        "enabled",
                        "Enable or disable follow-up replies",
                    )
                    .required(true),
                )
                .add_sub_option(CreateCommandOption::new(
                    CommandOptionType::Integer,
                    "timeout",
                    "Seconds to keep the conversation open without a ping (default 300)",
                )),
            );

        if let Err(e) = Command::create_global_command(&ctx.http, config_cmd).await {
            tracing::error!("Failed to register /config slash command: {e}");
        }
        let labs_cmd = CreateCommand::new("labs")
            .description("Enable experimental bot features")
            .add_option(CreateCommandOption::new(
                CommandOptionType::SubCommand,
                "list",
                "List experimental features and their status",
            ))
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "pagination",
                    "Toggle paginated LLM responses",
                )
                .add_sub_option(
                    CreateCommandOption::new(
                        CommandOptionType::Boolean,
                        "enabled",
                        "Enable or disable paginated responses",
                    )
                    .required(true),
                ),
            );
        if let Err(e) = Command::create_global_command(&ctx.http, labs_cmd).await {
            tracing::error!(target: "housebot::labs::registration", "Failed to register /labs slash command: {e}");
        }
        let mut effort_level_option = CreateCommandOption::new(
            CommandOptionType::String,
            "level",
            "Thinking effort level (omit to show the current setting)",
        );
        for mode in ThinkingMode::ALL {
            effort_level_option = effort_level_option
                .add_string_choice(format!("{mode} ({})", mode.budget_label()), mode.as_str());
        }
        let effort_cmd = CreateCommand::new("effort")
            .description("Set how much thinking the model does before replying")
            .add_option(effort_level_option);
        if let Err(e) = Command::create_global_command(&ctx.http, effort_cmd).await {
            tracing::error!("Failed to register /effort slash command: {e}");
        }
        let lua_cmd = CreateCommand::new("lua")
            .description("Run a sandboxed Lua script (requires the Scripting role)")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "script",
                    "Lua code to run (a ```lua code block``` is accepted)",
                )
                .required(true),
            );
        if let Err(e) = Command::create_global_command(&ctx.http, lua_cmd).await {
            tracing::error!("Failed to register /lua slash command: {e}");
        }
        for command in [
            CreateCommand::new("help").description("Show all available commands"),
            CreateCommand::new("commit").description("Show the bot's running commit hash"),
            CreateCommand::new("model").description("Show information about the current model"),
            CreateCommand::new("session")
                .description("Show context and token usage for this session"),
            CreateCommand::new("status")
                .description("Show your current settings (effort level, follow-up, personality)"),
            CreateCommand::new("new").description("Start a new conversation and clear the old one"),
            CreateCommand::new("reset").description("Clear the conversation and start fresh"),
            CreateCommand::new("compact")
                .description("Summarize the conversation and start a new session"),
            CreateCommand::new("erase_my_data").description(
                "Permanently delete all your stored data (messages, history, memory, notes)",
            ),
            CreateCommand::new("profile")
                .description("Show or clear your stored profile information")
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "show",
                    "Show your stored profile information",
                ))
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "clear",
                    "Clear learned profile data and memory",
                )),
            CreateCommand::new("history")
                .description("Show or clear your recent conversation history")
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "show",
                    "Show recent conversation history",
                ))
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "clear",
                    "Clear your conversation history",
                )),
            CreateCommand::new("privacy")
                .description("View or change your privacy settings")
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "status",
                    "Show current privacy settings",
                ))
                .add_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "deep_memory",
                        "Toggle deep memory",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::Boolean,
                            "enabled",
                            "Enable or disable deep memory",
                        )
                        .required(true),
                    ),
                )
                .add_option(
                    CreateCommandOption::new(
                        CommandOptionType::SubCommand,
                        "proactive",
                        "Toggle proactive assistance",
                    )
                    .add_sub_option(
                        CreateCommandOption::new(
                            CommandOptionType::Boolean,
                            "enabled",
                            "Enable or disable proactive assistance",
                        )
                        .required(true),
                    ),
                ),
            CreateCommand::new("memory")
                .description("View or clear the bot's persistent memory about you")
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "show",
                    "Show what the bot currently remembers about you",
                ))
                .add_option(CreateCommandOption::new(
                    CommandOptionType::SubCommand,
                    "clear",
                    "Clear the bot's memory about you",
                )),
        ] {
            if let Err(e) = Command::create_global_command(&ctx.http, command).await {
                tracing::error!("Failed to register slash command: {e}");
            }
        }

        if self.reminder_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let http = ctx.http.clone();
        let reminders = self.agent.reminders().clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let now = unix_now();
                for r in reminders.pop_due(now).await {
                    if let Ok(uid) = r.user_id.parse::<u64>() {
                        if let Ok(dm) = UserId::new(uid).create_dm_channel(&http).await {
                            let _ = dm
                                .say(&http, format!("⏰ **Reminder:** {}", r.message))
                                .await;
                        }
                    }
                }
            }
        });
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Component(component) = &interaction {
            if component.data.custom_id.starts_with(DEVELOP_PREFIX) {
                self.handle_develop_component(&ctx, component).await;
            } else {
                self.handle_pagination_component(&ctx, component).await;
            }
            return;
        }
        let Interaction::Command(cmd) = interaction else {
            return;
        };
        let user_id = cmd.user.id.get();
        let guild_id = cmd.guild_id.map(|g| g.get());
        tracing::info!(
            target: "housebot::commands",
            user_id,
            command = %cmd.data.name,
            "Slash command received"
        );
        if cmd.data.name == "compact" {
            let deep_memory_enabled = self.user_cfg.load(user_id).await.deep_memory_enabled;
            let response = CreateInteractionResponse::Defer(
                CreateInteractionResponseMessage::new().ephemeral(true),
            );
            if let Err(e) = cmd.create_response(&ctx.http, response).await {
                tracing::warn!("Failed to defer /compact response: {e}");
                return;
            }
            let hooks = CompactProgressHooks(CompactProgressTarget::Interaction {
                ctx: ctx.clone(),
                command: Box::new(cmd.clone()),
            });
            self.agent
                .compact_session_with_hooks(&user_id.to_string(), deep_memory_enabled, &hooks)
                .await;
            self.conversations
                .lock()
                .await
                .remove(cmd.channel_id.get(), user_id);
            return;
        }
        if cmd.data.name == "lua" {
            self.handle_lua_command(&ctx, &cmd).await;
            return;
        }
        let reply = match cmd.data.name.as_str() {
            "config" => {
                handle_config_interaction(
                    &self.server_cfg,
                    &self.user_cfg,
                    &cmd.data.options,
                    user_id,
                    guild_id,
                )
                .await
            }
            "labs" => handle_labs_interaction(&self.user_cfg, &cmd.data.options, user_id).await,
            "effort" => handle_effort_interaction(&self.user_cfg, &cmd.data.options, user_id).await,
            "status" => handle_status_interaction(&self.user_cfg, user_id).await,
            "help" => help_response(),
            "commit" => commit_hash_response(option_env!("HOUSEBOT_GIT_SHA")),
            "model" => self.agent.model_info(),
            "session" => {
                let info = self.agent.session_info(&user_id.to_string()).await;
                let percent =
                    info.context_tokens as f64 / info.context_window_tokens.max(1) as f64 * 100.0;
                let response = CreateInteractionResponse::Message(
                    CreateInteractionResponseMessage::new()
                        .embed(
                            CreateEmbed::new()
                                .title("Session")
                                .field(
                                    "Context",
                                    format!(
                                        "{} / {} tokens ({percent:.1}%)",
                                        info.context_tokens, info.context_window_tokens
                                    ),
                                    true,
                                )
                                .field("Messages", info.messages.to_string(), true)
                                .field("Model requests", info.requests.to_string(), true)
                                .field("Input tokens", info.input_tokens.to_string(), true)
                                .field("Output tokens", info.output_tokens.to_string(), true)
                                .field("Cached tokens", info.cached_tokens.to_string(), true),
                        )
                        .ephemeral(true),
                );
                if let Err(e) = cmd.create_response(&ctx.http, response).await {
                    tracing::warn!("Failed to send /session response: {e}");
                }
                return;
            }
            "new" | "reset" => self.handle_new(cmd.channel_id.get(), user_id).await,
            "erase_my_data" => {
                let reply = erase_data_command(
                    &self.message_log,
                    &self.history,
                    &self.memory,
                    &self.notes,
                    &self.profile_store,
                    &self.user_cfg,
                    &self.agent.reminders().clone(),
                    &self.channel_log,
                    user_id,
                )
                .await;
                self.agent.reset_session(&user_id.to_string()).await;
                self.conversations
                    .lock()
                    .await
                    .remove(cmd.channel_id.get(), user_id);
                reply
            }
            "profile" => {
                handle_profile_interaction(
                    &self.profile_store,
                    &self.memory,
                    &cmd.data.options,
                    user_id,
                    guild_id,
                )
                .await
            }
            "history" => {
                handle_history_interaction(
                    &self.history,
                    &self.profile_store,
                    &cmd.data.options,
                    user_id,
                    guild_id,
                )
                .await
            }
            "privacy" => {
                handle_privacy_interaction(&self.user_cfg, &self.memory, &cmd.data.options, user_id)
                    .await
            }
            "memory" => handle_memory_interaction(&self.memory, &cmd.data.options, user_id).await,
            _ => return,
        };

        let response = CreateInteractionResponse::Message(
            CreateInteractionResponseMessage::new()
                .content(reply)
                .ephemeral(true),
        );
        if let Err(e) = cmd.create_response(&ctx.http, response).await {
            tracing::warn!("Failed to send /config response: {e}");
        }
    }

    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }
        let content = msg.content.trim().to_string();
        let channel_id = msg.channel_id.get();
        let user_id = msg.author.id.get();

        // ── commands ──
        if content == "!reset" || content == "!new" {
            let reply = self.handle_new(channel_id, user_id).await;
            self.respond(&ctx, &msg, &reply).await;
            return;
        }
        if content == "!compact" {
            let deep_memory_enabled = self.user_cfg.load(user_id).await.deep_memory_enabled;
            let progress = reply_no_ping(&ctx, &msg, &compact_progress(0, None))
                .await
                .ok();
            if let Some(progress) = &progress {
                let hooks = CompactProgressHooks(CompactProgressTarget::Message {
                    ctx: ctx.clone(),
                    channel_id: msg.channel_id,
                    message_id: progress.id,
                });
                self.agent
                    .compact_session_with_hooks(&user_id.to_string(), deep_memory_enabled, &hooks)
                    .await;
            } else {
                self.agent
                    .compact_session(&user_id.to_string(), deep_memory_enabled)
                    .await;
            }
            self.conversations.lock().await.remove(channel_id, user_id);
            if let Some(mut progress) = progress {
                let _ = progress
                    .edit(
                        &ctx.http,
                        EditMessage::new().content(compact_done_message(deep_memory_enabled)),
                    )
                    .await;
            } else {
                self.respond(&ctx, &msg, compact_done_message(deep_memory_enabled))
                    .await;
            }
            return;
        }
        if msg.content.starts_with("!skill") {
            tracing::info!(target: "housebot::commands", user_id, "!skill command received");
            let (first, rest) = split_command(&msg.content);
            let reply = skill_command(&self.skills, &first, &rest, user_id).await;
            self.respond(&ctx, &msg, &reply).await;
            return;
        }
        if msg.content.starts_with("!note") {
            tracing::info!(target: "housebot::commands", user_id, "!note command received");
            let (first, rest) = split_command(&msg.content);
            let reply = note_command(&self.notes, &first, &rest, user_id).await;
            self.respond(&ctx, &msg, &reply).await;
            return;
        }
        if content == "!stats" {
            let reply = stats_command(
                &self.history,
                &self.memory,
                &self.notes,
                &self.skills,
                user_id,
                &msg.author.name,
            )
            .await;
            self.respond(&ctx, &msg, &reply).await;
            return;
        }
        // ── routing ──
        let bot_id = ctx.cache.current_user().id;
        let is_dm = msg.guild_id.is_none();
        let guild_id = msg.guild_id.map(|g| g.get());

        // Check channel allowlist before doing anything else.
        if !self
            .server_cfg
            .is_channel_allowed(guild_id, channel_id)
            .await
        {
            return;
        }

        if !is_dm {
            // Prefer server nickname, then global display name, over the raw username.
            let nick = msg
                .member
                .as_ref()
                .and_then(|m| m.nick.as_deref())
                .or(msg.author.global_name.as_deref())
                .filter(|n| *n != msg.author.name);
            self.channel_log
                .append(channel_id, user_id, &msg.author.name, nick, &content)
                .await;
        }

        let is_mentioned = msg.mentions.iter().any(|u| u.id == bot_id);
        let is_reply_to_bot = msg
            .referenced_message
            .as_ref()
            .map(|m| m.author.id == bot_id)
            .unwrap_or(false);
        let is_reply_to_media = msg
            .referenced_message
            .as_deref()
            .is_some_and(message_has_supported_media);

        // Follow-ups are on by default in DMs. In guild channels, users must
        // opt in and the channel must be explicitly configured by the server.
        let user_config = self.user_cfg.load(user_id).await;
        let followup_enabled = is_dm || user_config.followup_enabled;
        let followup_timeout = Duration::from_secs(user_config.followup_timeout_secs);
        let followup_channel_allowed = self
            .server_cfg
            .is_followup_channel_allowed(guild_id, channel_id)
            .await;
        let followup_channel_allowed = is_dm || followup_channel_allowed;

        let now = Instant::now();
        let (is_active, session_expired) = {
            let mut convos = self.conversations.lock().await;
            let active = followup_enabled
                && followup_channel_allowed
                && convos.is_active(channel_id, user_id, now);
            let expired = !active && convos.pop_timed_out(channel_id, user_id, now);
            (active, expired)
        };

        let proactive = !is_dm
            && user_config.proactive_assistance_enabled
            && !is_mentioned
            && !is_reply_to_bot
            && !is_reply_to_media
            && is_proactive_candidate(&content)
            && self.proactive_cooldown_allows(channel_id, user_id).await;
        if !(is_dm
            || is_mentioned
            || is_reply_to_bot
            || is_reply_to_media
            || is_active
            || proactive)
        {
            return;
        }
        if self.already_seen(msg.id.get()).await {
            tracing::warn!("Duplicate message {} — skipping", msg.id.get());
            return;
        }

        self.handle_message(
            &ctx,
            &msg,
            bot_id,
            session_expired,
            followup_timeout,
            proactive,
        )
        .await;
        self.mark_done(msg.id.get()).await;
    }
}

impl HouseBot {
    /// Handle the `/lua` slash command: permission and rate checks, then run
    /// the script in the sandbox and post its output.
    async fn handle_lua_command(&self, ctx: &Context, cmd: &serenity::all::CommandInteraction) {
        let user_id = cmd.user.id.get();
        let script = cmd
            .data
            .options
            .iter()
            .find(|o| o.name == "script")
            .and_then(|o| match &o.value {
                CommandDataOptionValue::String(s) => Some(s.clone()),
                _ => None,
            });
        let Some(script) = script else {
            respond_ephemeral(ctx, cmd, "Please provide a script to run.").await;
            return;
        };
        if !self.lua_permitted(ctx, cmd).await {
            let reply = format!(
                "You need the **{}** role (or a higher one) to run scripts.",
                lua_engine::scripting_role_name()
            );
            respond_ephemeral(ctx, cmd, &reply).await;
            return;
        }
        if self.lua_rate_limiter.check(&user_id.to_string()) {
            respond_ephemeral(
                ctx,
                cmd,
                "You're running scripts too quickly — try again in a minute.",
            )
            .await;
            return;
        }
        tracing::info!(target: "housebot::commands", user_id, "Running /lua script");
        let defer = CreateInteractionResponse::Defer(CreateInteractionResponseMessage::new());
        if let Err(e) = cmd.create_response(&ctx.http, defer).await {
            tracing::warn!("Failed to defer /lua response: {e}");
            return;
        }
        let host = Arc::new(lua_engine::BotScriptHost {
            agent: Arc::clone(&self.agent),
            discord: Arc::clone(&self.discord),
            channel_id: cmd.channel_id.get(),
        });
        let script = lua_engine::strip_code_fence(&script).to_string();
        let output = lua_engine::run_script(script, host, lua_engine::LuaLimits::from_env()).await;
        let reply = format_lua_reply(&self.redactor.redact(&output));
        if let Err(e) = cmd
            .edit_response(&ctx.http, EditInteractionResponse::new().content(reply))
            .await
        {
            tracing::warn!("Failed to send /lua response: {e}");
        }
    }

    /// `/lua` is allowed for the bot owner, guild administrators, and members
    /// holding the scripting role or a higher one.
    async fn lua_permitted(&self, ctx: &Context, cmd: &serenity::all::CommandInteraction) -> bool {
        let user_id = cmd.user.id.get();
        let owner_id = config::owner_id();
        if owner_id != 0 && user_id == owner_id {
            return true;
        }
        let (Some(guild_id), Some(member)) = (cmd.guild_id, cmd.member.as_deref()) else {
            return false;
        };
        if member.permissions.is_some_and(|p| p.administrator()) {
            return true;
        }
        let Ok(roles) = guild_id.roles(&ctx.http).await else {
            return false;
        };
        let guild_roles: Vec<(u64, String, u16)> = roles
            .values()
            .map(|role| (role.id.get(), role.name.clone(), role.position))
            .collect();
        let member_roles: Vec<u64> = member.roles.iter().map(|r| r.get()).collect();
        lua_engine::scripting_permitted(
            &member_roles,
            &guild_roles,
            &lua_engine::scripting_role_name(),
        )
    }

    async fn handle_message(
        &self,
        ctx: &Context,
        msg: &Message,
        bot_id: UserId,
        session_expired: bool,
        followup_timeout: Duration,
        proactive: bool,
    ) {
        let mut text = msg.content.clone();
        for token in [format!("<@{bot_id}>"), format!("<@!{bot_id}>")] {
            text = text.replace(&token, "");
        }
        let text = text.trim().to_string();

        let referenced_text = msg
            .referenced_message
            .as_deref()
            .and_then(referenced_message_context);
        let text = match referenced_text {
            Some(referenced) if text.is_empty() => referenced,
            Some(referenced) => format!("{text}\n\n{referenced}"),
            None => text,
        };
        if text.is_empty() && !message_has_supported_media(msg) {
            return;
        }

        if self
            .chat_rate_limiter
            .check(&msg.author.id.get().to_string())
        {
            tracing::warn!(
                target: "housebot::rate_limit",
                user_id = msg.author.id.get(),
                "Chat rate limit exceeded"
            );
            self.respond(ctx, msg, "⏱️ You're sending messages too quickly. Please slow down and try again in a moment.").await;
            return;
        }

        let user_config = self.user_cfg.load(msg.author.id.get()).await;

        if session_expired {
            self.agent
                .compact_session(
                    &msg.author.id.get().to_string(),
                    user_config.deep_memory_enabled,
                )
                .await;
        }

        let mut media = extract_media(msg).await;
        if let Some(referenced) = msg.referenced_message.as_deref() {
            media.extend(extract_media(referenced).await);
        }

        // Load per-user settings (personality, thinking effort, and privacy).
        let personality = user_config.personality.clone();
        let thinking = user_config.thinking_mode;

        // Refresh user profile from Discord and persist learned data.
        let mut profile = self.profile_store.load(msg.author.id.get()).await;
        let guild_id = msg.guild_id.map(|g| g.get()).unwrap_or(0);
        if profile.username.is_empty() || profile.guild_id != guild_id {
            // First time seeing this user in this guild — fetch profile from Discord.
            if let Ok(user_info) = self.discord.fetch_user(msg.author.id.get()).await {
                profile.username = user_info.username;
                profile.display_name = user_info.display_name;
                profile.avatar_url = user_info.avatar_url.unwrap_or_default();
                profile.guild_id = guild_id;
                profile.nickname.clear();
                if let Some(guild) = msg.guild(&ctx.cache) {
                    if let Some(member) = guild.members.get(&msg.author.id) {
                        if let Some(nick) = &member.nick {
                            profile.nickname = nick.clone();
                        }
                    }
                }
                let _ = self.profile_store.save(msg.author.id.get(), &profile).await;
            }
        } else {
            // Update display name and nickname if they've changed.
            if let Ok(user_info) = self.discord.fetch_user(msg.author.id.get()).await {
                if profile.display_name != user_info.display_name {
                    profile.display_name = user_info.display_name;
                }
                let avatar = user_info.avatar_url.clone().unwrap_or_default();
                if profile.avatar_url != avatar {
                    profile.avatar_url = avatar;
                }
                if let Some(guild) = msg.guild(&ctx.cache) {
                    if let Some(member) = guild.members.get(&msg.author.id) {
                        let current_nick = member.nick.as_deref().unwrap_or("");
                        if profile.nickname != current_nick {
                            profile.nickname = current_nick.to_string();
                        }
                    }
                }
                let _ = self.profile_store.save(msg.author.id.get(), &profile).await;
            }
        }

        // Keep the progress message in the thinking state until the model starts its final reply.
        let progress = reply_no_ping(ctx, msg, "🧠 **Thinking...**").await.ok();
        let pending_reaction = msg.react(&ctx.http, '⏳').await.ok();

        let response_hooks = progress
            .as_ref()
            .map(|progress| ResponseProgressHooks::new(ctx, progress));

        let user_text = if text.is_empty() {
            "(no text)".to_string()
        } else {
            text
        };
        self.message_log
            .append(msg.author.id.get().to_string(), &user_text)
            .await;
        let user_id_string = msg.author.id.get().to_string();
        let profile_tags = profile
            .tags
            .iter()
            .map(|tag| tag.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let quick_actions = profile
            .quick_actions()
            .into_iter()
            .map(|(name, count)| format!("{name} ({count})"))
            .collect::<Vec<_>>()
            .join(", ");
        let result: AgentResult = self
            .agent
            .run(
                AgentRequest {
                    user_id: &user_id_string,
                    username: &msg.author.name,
                    text: &user_text,
                    media: &media,
                    personality: personality.as_deref(),
                    thinking,
                    channel_id: msg.channel_id.get(),
                    deep_memory_enabled: user_config.deep_memory_enabled && !proactive,
                    display_name: &profile.display_name,
                    nickname: &profile.nickname,
                    avatar_url: &profile.avatar_url,
                    profile_tags: &profile_tags,
                    quick_actions: &quick_actions,
                    guild_id: msg.guild_id.map(|guild| guild.get()),
                    proactive,
                    record_profile_usage: !proactive,
                },
                response_hooks
                    .as_ref()
                    .map_or(&NoHooks as &dyn AgentHooks, |hooks| {
                        hooks as &dyn AgentHooks
                    }),
            )
            .await;

        {
            let mut convos = self.conversations.lock().await;
            convos.mark_active(
                msg.channel_id.get(),
                msg.author.id.get(),
                Instant::now(),
                followup_timeout,
            );
        }

        // Handle structured development control actions before displaying text.
        if let Some(action) = result.control_action {
            if let Some(reaction) = pending_reaction {
                let _ = reaction.delete(&ctx.http).await;
            }
            if let Some(progress) = progress.as_ref() {
                let _ = progress.delete(&ctx.http).await;
            }
            match action {
                AgentControlAction::OwnerDispatchReady { job_id } => {
                    self.dispatch_owner_job_immediately(ctx, msg, job_id).await;
                }
                AgentControlAction::OwnerConfigurationRequired { job_id } => {
                    self.start_develop_flow(ctx, msg, job_id).await;
                }
                AgentControlAction::OwnerApprovalRequired { job_id } => {
                    // Reply to requester, then DM the owner.
                    self.respond(
                        ctx,
                        msg,
                        "I sent this development request to the bot owner for approval. \
                         Work will not start unless the owner approves it.",
                    )
                    .await;
                    self.notify_owner_for_approval(ctx, msg, job_id).await;
                }
            }
            return;
        }

        let safe = self.redactor.redact(&result.text);
        if let Some(notice) = &result.session_notice {
            let _ = reply_no_ping(ctx, msg, notice).await;
        }
        let with_tool_summary = append_tool_summary(&safe, &result.tools_called);
        let (display, code_files) = extract_code_files(&with_tool_summary);
        send_final_message(
            ctx,
            msg,
            &display,
            user_config.labs_pagination_enabled,
            msg.author.id.get(),
            &self.paginated,
            progress.as_ref(),
        )
        .await;

        if let Some(reaction) = pending_reaction {
            let _ = reaction.delete(&ctx.http).await;
        }
        // Upload extracted code blocks.
        for (filename, content) in code_files {
            let safe = self.redactor.redact(&String::from_utf8_lossy(&content));
            let _ = msg
                .channel_id
                .send_message(
                    &ctx.http,
                    CreateMessage::new()
                        .add_file(CreateAttachment::bytes(safe.into_bytes(), filename)),
                )
                .await;
        }
    }

    /// Send the initial agent-selection message for an interactive develop job.
    async fn start_develop_flow(&self, ctx: &Context, msg: &Message, job_id: Uuid) {
        let title = self
            .pending_jobs
            .with_job(job_id, |j| j.specification.title.clone());
        let Some(title) = title else {
            let _ = reply_no_ping(ctx, msg, "Error: Development job not found.").await;
            return;
        };
        let content = format!(
            "**Feature development: {title}**\n\nChoose a coding agent to implement this feature:"
        );
        let components = develop_agent_components(&job_id.to_string());
        let builder = CreateMessage::new()
            .content(content)
            .components(components)
            .reference_message(msg)
            .allowed_mentions(CreateAllowedMentions::new());
        if let Ok(sent) = msg.channel_id.send_message(&ctx.http, builder).await {
            self.pending_jobs.with_job_mut(job_id, |j| {
                j.approval_message = Some(DiscordMessageRef {
                    channel_id: sent.channel_id.get(),
                    message_id: sent.id.get(),
                });
            });
        }
    }

    /// Immediately dispatch an owner-direct job without interactive confirmation.
    async fn dispatch_owner_job_immediately(&self, ctx: &Context, msg: &Message, job_id: Uuid) {
        // Atomically transition Confirming → Dispatching.
        if !self.pending_jobs.try_start_dispatch(job_id) {
            let _ = reply_no_ping(
                ctx,
                msg,
                "❌ Failed to dispatch: job is not in a dispatchable state.",
            )
            .await;
            return;
        }

        let job_data = self.pending_jobs.with_job(job_id, |j| {
            let agent = j.selection.agent?;
            let model = j.selection.model.clone()?;
            let effort = j.selection.effort.clone()?;
            Some((
                j.specification.clone(),
                agent,
                model,
                effort,
                j.requester.username.clone(),
                j.requester.user_id,
            ))
        });
        let Some(Some((spec, agent, model, effort, req_name, req_id))) = job_data else {
            self.pending_jobs.mark_dispatch_failed(job_id);
            let _ = reply_no_ping(
                ctx,
                msg,
                "❌ Failed to dispatch: incomplete agent/model/effort selection. \
                 Please set DEVELOPMENT_DEFAULT_AGENT, DEVELOPMENT_DEFAULT_MODEL, \
                 and DEVELOPMENT_DEFAULT_EFFORT, or use the interactive flow.",
            )
            .await;
            return;
        };

        let selection = match self.catalog.validate_selection(agent, &model, &effort) {
            Ok(s) => s,
            Err(e) => {
                self.pending_jobs.mark_dispatch_failed(job_id);
                let _ = reply_no_ping(ctx, msg, &format!("❌ Configuration error: {e}")).await;
                return;
            }
        };

        let body = match build_issue_body(&spec, &selection, &req_name, req_id, &req_name, req_id) {
            Ok(b) => b,
            Err(e) => {
                self.pending_jobs.mark_dispatch_failed(job_id);
                let _ =
                    reply_no_ping(ctx, msg, &format!("❌ Failed to build issue body: {e}")).await;
                return;
            }
        };

        let title = format!("[agent:{}] {}", agent.id_str(), spec.title);
        let labels = dispatch_labels(agent);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let reporter = self.agent.reporter();
        match reporter.create_issue_full(&title, &body, &label_refs).await {
            Some(issue) => {
                self.pending_jobs.mark_dispatched(job_id);
                tracing::info!(
                    target: "housebot::develop",
                    issue_number = issue.number,
                    agent = agent.id_str(),
                    "Owner-immediate development job dispatched"
                );
                let _ = reply_no_ping(
                    ctx,
                    msg,
                    &format!(
                        "✅ **Dispatched!**\n\
                         Issue #{num} created: {url}\n\
                         Agent: **{agent_name}** | Model: `{model}` | Effort: `{effort}`\n\
                         The GitHub Actions workflow will pick this up and open a draft PR.",
                        num = issue.number,
                        url = issue.html_url,
                        agent_name = agent.display_name(),
                    ),
                )
                .await;
            }
            None => {
                self.pending_jobs.mark_dispatch_failed(job_id);
                let _ = reply_no_ping(
                    ctx,
                    msg,
                    "❌ Failed to create GitHub issue. Check bot logs for details.",
                )
                .await;
            }
        }
    }

    /// DM the configured owner about a non-owner approval request.
    async fn notify_owner_for_approval(
        &self,
        ctx: &Context,
        requester_msg: &Message,
        job_id: Uuid,
    ) {
        let owner_id = config::owner_id();
        if owner_id == 0 {
            tracing::warn!(target: "housebot::develop", "Cannot notify owner: OWNER_DISCORD_ID not set");
            return;
        }

        let job_info = self.pending_jobs.with_job(job_id, |j| {
            (
                j.specification.title.clone(),
                j.specification.objective.clone(),
                j.requester.username.clone(),
                j.requester.user_id,
                j.requester.channel_id,
                j.selection.agent,
                j.selection.model.clone(),
                j.selection.effort.clone(),
            )
        });
        let Some((title, objective, req_name, req_id, req_channel, agent, model, effort)) =
            job_info
        else {
            tracing::warn!(target: "housebot::develop", %job_id, "Job not found when notifying owner");
            return;
        };

        let agent_str = agent
            .map(|a| a.display_name().to_string())
            .unwrap_or_else(|| "default".into());
        let model_str = model.as_deref().unwrap_or("default");
        let effort_str = effort.as_deref().unwrap_or("default");

        let dm_content = format!(
            "**Feature-development request from <@{req_id}>** (`{req_name}`)\n\
             **Feature:** {title}\n\
             **Objective:**\n> {obj}\n\
             **Proposed configuration:**\n\
             Agent: {agent_str} | Model: `{model_str}` | Effort: `{effort_str}`\n\
             **Origin:** <#{req_channel}>",
            obj = objective.lines().collect::<Vec<_>>().join("\n> "),
        );

        let id_str = job_id.to_string();
        let components = develop_approval_components(&id_str);

        let send_dm = async {
            let owner_user = UserId::new(owner_id).to_user(&ctx.http).await?;
            let dm = owner_user.create_dm_channel(&ctx.http).await?;
            let builder = CreateMessage::new()
                .content(&dm_content)
                .components(components.clone());
            dm.send_message(&ctx.http, builder).await
        };

        match send_dm.await {
            Ok(sent) => {
                self.pending_jobs.with_job_mut(job_id, |j| {
                    j.approval_message = Some(DiscordMessageRef {
                        channel_id: sent.channel_id.get(),
                        message_id: sent.id.get(),
                    });
                });
                tracing::info!(
                    target: "housebot::develop",
                    %job_id,
                    requester_id = req_id,
                    "Owner DM sent for approval"
                );
            }
            Err(e) => {
                tracing::error!(
                    target: "housebot::develop",
                    %job_id,
                    error = %e,
                    "Failed to DM owner for approval"
                );
                // Try fallback channel.
                let fallback =
                    crate::config::env_parse::<u64>("DEVELOPMENT_APPROVAL_CHANNEL_ID", 0);
                if fallback != 0 {
                    let fb_channel = serenity::all::ChannelId::new(fallback);
                    let builder = CreateMessage::new()
                        .content(&dm_content)
                        .components(components);
                    if let Ok(sent) = fb_channel.send_message(&ctx.http, builder).await {
                        self.pending_jobs.with_job_mut(job_id, |j| {
                            j.approval_message = Some(DiscordMessageRef {
                                channel_id: sent.channel_id.get(),
                                message_id: sent.id.get(),
                            });
                        });
                        tracing::info!(
                            target: "housebot::develop",
                            %job_id,
                            "Approval card sent to fallback channel"
                        );
                        return;
                    }
                }
                // Both DM and fallback failed — cancel the job so it doesn't accumulate invisibly.
                self.pending_jobs.cancel(job_id);
                self.respond(
                    ctx,
                    requester_msg,
                    "I prepared the request, but I could not contact the owner for approval.",
                )
                .await;
            }
        }
    }

    /// Handle a Discord component interaction for the develop flow.
    async fn handle_develop_component(
        &self,
        ctx: &Context,
        component: &serenity::all::ComponentInteraction,
    ) {
        // custom_id format: develop:<job-id>:<action>
        let rest = component
            .data
            .custom_id
            .strip_prefix(DEVELOP_PREFIX)
            .unwrap_or("");
        let Some((id_str, action)) = rest.split_once(':') else {
            return;
        };
        let Ok(job_id) = id_str.parse::<Uuid>() else {
            return;
        };

        let owner_id = self.pending_jobs.with_job(job_id, |j| j.owner_id);
        let Some(owner_id) = owner_id else {
            let _ = component
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content(
                                "This development job has expired. Please ask the bot to prepare a new one.",
                            )
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        };

        // Only the owner may interact.
        if component.user.id.get() != owner_id {
            let _ = component
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Only the configured bot owner can use these controls.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }

        // Check expiry.
        let expired = self
            .pending_jobs
            .with_job(job_id, |j| j.is_expired())
            .unwrap_or(true);
        if expired {
            let _ = component
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content(
                                "This development job has expired (15-minute timeout). Please ask the bot to prepare a new one.",
                            )
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }

        let id_str = job_id.to_string();
        match action {
            "agent" => {
                // Value from the select menu.
                let selected = match &component.data.kind {
                    ComponentInteractionDataKind::StringSelect { values } => {
                        values.first().cloned()
                    }
                    _ => None,
                };
                let Some(agent_id) = selected else {
                    return;
                };
                let Ok(agent) = agent_id.parse::<CodingAgent>() else {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content(format!("Unknown agent: {agent_id}"))
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                };
                self.pending_jobs.with_job_mut(job_id, |j| {
                    j.selection.agent = Some(agent);
                    j.selection.model = None;
                    j.selection.effort = None;
                    j.stage = DispatchStage::ChoosingModel;
                });
                let (title, models_text) = self
                    .pending_jobs
                    .with_job(job_id, |j| {
                        (
                            j.specification.title.clone(),
                            format!(
                                "**Feature development: {}**\n\n\
                                 Agent: **{}**\nChoose a model:",
                                j.specification.title,
                                agent.display_name()
                            ),
                        )
                    })
                    .unwrap_or_default();
                let _ = title;
                let components = develop_model_components(&id_str, agent, &self.catalog);
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content(models_text)
                                .components(components),
                        ),
                    )
                    .await;
            }
            "model" => {
                let selected = match &component.data.kind {
                    ComponentInteractionDataKind::StringSelect { values } => {
                        values.first().cloned()
                    }
                    _ => None,
                };
                let Some(model_id) = selected else {
                    return;
                };
                let agent = self
                    .pending_jobs
                    .with_job(job_id, |j| j.selection.agent)
                    .flatten();
                let Some(agent) = agent else {
                    return;
                };
                // Validate model against catalog.
                if self.catalog.efforts_for(agent, &model_id).is_none() {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content(format!(
                                        "Model `{model_id}` is not valid for {agent}."
                                    ))
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                }
                self.pending_jobs.with_job_mut(job_id, |j| {
                    j.selection.model = Some(model_id.clone());
                    j.selection.effort = None;
                    j.stage = DispatchStage::ChoosingEffort;
                });
                let content = self
                    .pending_jobs
                    .with_job(job_id, |j| {
                        format!(
                            "**Feature development: {}**\n\n\
                             Agent: **{}**\nModel: **{}**\nChoose effort level:",
                            j.specification.title,
                            agent.display_name(),
                            model_id
                        )
                    })
                    .unwrap_or_default();
                let components =
                    develop_effort_components(&id_str, agent, &model_id, &self.catalog);
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content(content)
                                .components(components),
                        ),
                    )
                    .await;
            }
            "effort" => {
                let selected = match &component.data.kind {
                    ComponentInteractionDataKind::StringSelect { values } => {
                        values.first().cloned()
                    }
                    _ => None,
                };
                let Some(effort_id) = selected else {
                    return;
                };
                let (agent, model) = self
                    .pending_jobs
                    .with_job(job_id, |j| (j.selection.agent, j.selection.model.clone()))
                    .unwrap_or_default();
                let (Some(agent), Some(model)) = (agent, model) else {
                    return;
                };
                // Validate effort.
                if self
                    .catalog
                    .efforts_for(agent, &model)
                    .and_then(|efs| efs.iter().find(|e| e.id == effort_id))
                    .is_none()
                {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content(format!(
                                        "Effort `{effort_id}` is not valid for model `{model}`."
                                    ))
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                }
                self.pending_jobs.with_job_mut(job_id, |j| {
                    j.selection.effort = Some(effort_id.clone());
                    j.stage = DispatchStage::Confirming;
                });
                let content = self
                    .pending_jobs
                    .with_job(job_id, |j| {
                        format!(
                            "**Feature development: {}**\n\n\
                             **Agent:** {}\n\
                             **Model:** {}\n\
                             **Effort:** {}\n\n\
                             **Objective:**\n{}\n\n\
                             Confirm dispatch to create a GitHub issue and queue the coding job.",
                            j.specification.title,
                            agent.display_name(),
                            model,
                            effort_id,
                            j.specification.objective
                        )
                    })
                    .unwrap_or_default();
                let components = develop_confirm_components(&id_str);
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content(content)
                                .components(components),
                        ),
                    )
                    .await;
            }
            "confirm" => {
                // Atomic dispatch: only succeeds once.
                if !self.pending_jobs.try_start_dispatch(job_id) {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content(
                                        "This job is already being dispatched or has been dispatched.",
                                    )
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                }

                // Acknowledge immediately so Discord doesn't timeout.
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content("⏳ **Dispatching...** Creating GitHub issue...")
                                .components(vec![]),
                        ),
                    )
                    .await;

                // Gather all needed data — use original requester, not the approver.
                let job_data = self.pending_jobs.with_job(job_id, |j| {
                    let agent = j.selection.agent?;
                    let model = j.selection.model.clone()?;
                    let effort = j.selection.effort.clone()?;
                    Some((
                        j.specification.clone(),
                        agent,
                        model,
                        effort,
                        j.requester.username.clone(),
                        j.requester.user_id,
                    ))
                });
                let Some(Some((spec, agent, model, effort, requester_name, requester_user_id))) =
                    job_data
                else {
                    self.pending_jobs.mark_dispatch_failed(job_id);
                    let _ = component
                        .edit_response(
                            &ctx.http,
                            EditInteractionResponse::new().content(
                                "❌ Failed to dispatch: incomplete selection. Please start again.",
                            ),
                        )
                        .await;
                    return;
                };

                let selection = match self.catalog.validate_selection(agent, &model, &effort) {
                    Ok(s) => s,
                    Err(e) => {
                        self.pending_jobs.mark_dispatch_failed(job_id);
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new().content(format!(
                                    "❌ Configuration error: {e}. Please start again."
                                )),
                            )
                            .await;
                        return;
                    }
                };

                let approver_name = component.user.name.clone();
                let approver_id = component.user.id.get();

                let body = match build_issue_body(
                    &spec,
                    &selection,
                    &requester_name,
                    requester_user_id,
                    &approver_name,
                    approver_id,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        self.pending_jobs.mark_dispatch_failed(job_id);
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new()
                                    .content(format!("❌ Failed to build issue body: {e}")),
                            )
                            .await;
                        return;
                    }
                };

                let title = format!("[agent:{}] {}", agent.id_str(), spec.title);
                let labels = dispatch_labels(agent);
                let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

                // Get the reporter from the agent.
                let reporter = self.agent.reporter();
                match reporter.create_issue_full(&title, &body, &label_refs).await {
                    Some(issue) => {
                        self.pending_jobs.mark_dispatched(job_id);
                        tracing::info!(
                            target: "housebot::develop",
                            issue_number = issue.number,
                            agent = agent.id_str(),
                            "Development job dispatched"
                        );
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new().content(format!(
                                    "✅ **Dispatched!**\n\
                                     Issue #{num} created: {url}\n\
                                     Agent: **{agent_name}** | Model: `{model}` | Effort: `{effort}`\n\
                                     The GitHub Actions workflow will pick this up and open a draft PR.",
                                    num = issue.number,
                                    url = issue.html_url,
                                    agent_name = agent.display_name(),
                                )),
                            )
                            .await;
                    }
                    None => {
                        self.pending_jobs.mark_dispatch_failed(job_id);
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new().content(
                                    "❌ Failed to create GitHub issue. Check bot logs for details.\n\
                                     The job has been reset to the confirmation stage — click Dispatch to retry.",
                                ),
                            )
                            .await;
                    }
                }
            }
            "approve" => {
                // Owner approves a non-owner request with default selection.
                if !self.pending_jobs.try_approve_with_defaults(job_id) {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content(
                                        "This request cannot be approved now (wrong stage, expired, or selection incomplete).",
                                    )
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                }

                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content("⏳ **Approving...** Creating GitHub issue...")
                                .components(vec![]),
                        ),
                    )
                    .await;

                let job_data = self.pending_jobs.with_job(job_id, |j| {
                    let agent = j.selection.agent?;
                    let model = j.selection.model.clone()?;
                    let effort = j.selection.effort.clone()?;
                    Some((
                        j.specification.clone(),
                        agent,
                        model,
                        effort,
                        j.requester.username.clone(),
                        j.requester.user_id,
                        j.requester.channel_id,
                    ))
                });
                let Some(Some((spec, agent, model, effort, req_name, req_id, req_channel))) =
                    job_data
                else {
                    self.pending_jobs.mark_dispatch_failed(job_id);
                    let _ = component
                        .edit_response(
                            &ctx.http,
                            EditInteractionResponse::new()
                                .content("❌ Failed: incomplete selection."),
                        )
                        .await;
                    return;
                };

                let selection = match self.catalog.validate_selection(agent, &model, &effort) {
                    Ok(s) => s,
                    Err(e) => {
                        self.pending_jobs.mark_dispatch_failed(job_id);
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new()
                                    .content(format!("❌ Configuration error: {e}")),
                            )
                            .await;
                        return;
                    }
                };

                let approver_name = component.user.name.clone();
                let approver_id = component.user.id.get();
                let body = match build_issue_body(
                    &spec,
                    &selection,
                    &req_name,
                    req_id,
                    &approver_name,
                    approver_id,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        self.pending_jobs.mark_dispatch_failed(job_id);
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new()
                                    .content(format!("❌ Failed to build issue body: {e}")),
                            )
                            .await;
                        return;
                    }
                };

                let title = format!("[agent:{}] {}", agent.id_str(), spec.title);
                let labels = dispatch_labels(agent);
                let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                let reporter = self.agent.reporter();
                match reporter.create_issue_full(&title, &body, &label_refs).await {
                    Some(issue) => {
                        self.pending_jobs.mark_dispatched(job_id);
                        tracing::info!(
                            target: "housebot::develop",
                            issue_number = issue.number,
                            agent = agent.id_str(),
                            "Non-owner development job approved and dispatched"
                        );
                        let success_msg = format!(
                            "✅ **Dispatched!**\n\
                             Issue #{num} created: {url}\n\
                             Agent: **{agent_name}** | Model: `{model}` | Effort: `{effort}`\n\
                             The GitHub Actions workflow will pick this up and open a draft PR.",
                            num = issue.number,
                            url = issue.html_url,
                            agent_name = agent.display_name(),
                        );
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new().content(&success_msg),
                            )
                            .await;
                        // Notify original requester.
                        let channel = serenity::all::ChannelId::new(req_channel);
                        let _ = channel
                            .say(
                                &ctx.http,
                                format!(
                                    "✅ <@{req_id}> The bot owner approved your development request. \
                                     Development has started using {agent_name}, `{model}`, `{effort}`.\n\
                                     Issue: {url}",
                                    agent_name = agent.display_name(),
                                    url = issue.html_url,
                                ),
                            )
                            .await;
                    }
                    None => {
                        self.pending_jobs.mark_dispatch_failed(job_id);
                        let _ = component
                            .edit_response(
                                &ctx.http,
                                EditInteractionResponse::new().content(
                                    "❌ Failed to create GitHub issue. The job has been reset — click Start Work to retry.",
                                ),
                            )
                            .await;
                    }
                }
            }
            "configure" => {
                // Owner wants to change agent/model/effort before approving.
                if !self.pending_jobs.try_begin_configuration(job_id) {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content("Cannot begin configuration from the current state.")
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                }
                let title = self
                    .pending_jobs
                    .with_job(job_id, |j| j.specification.title.clone())
                    .unwrap_or_default();
                let content = format!(
                    "**Feature development: {title}**\n\nChoose a coding agent to implement this feature:"
                );
                let components = develop_agent_components(&id_str);
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content(content)
                                .components(components),
                        ),
                    )
                    .await;
            }
            "reject" => {
                let req_channel = self
                    .pending_jobs
                    .with_job(job_id, |j| (j.requester.channel_id, j.requester.user_id));
                if !self.pending_jobs.try_reject(job_id) {
                    let _ = component
                        .create_response(
                            &ctx.http,
                            CreateInteractionResponse::Message(
                                CreateInteractionResponseMessage::new()
                                    .content("This request is no longer active.")
                                    .ephemeral(true),
                            ),
                        )
                        .await;
                    return;
                }
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content("❌ Request rejected.")
                                .components(vec![]),
                        ),
                    )
                    .await;
                // Notify requester.
                if let Some((channel_id, requester_id)) = req_channel {
                    let channel = serenity::all::ChannelId::new(channel_id);
                    let _ = channel
                        .say(
                            &ctx.http,
                            format!(
                                "<@{requester_id}> Your automated development request was not approved by the bot owner."
                            ),
                        )
                        .await;
                }
            }
            "back" => {
                // Navigate back one stage.
                let stage = self.pending_jobs.with_job(job_id, |j| j.stage);
                let (content, components) = match stage {
                    Some(DispatchStage::ChoosingModel) => {
                        self.pending_jobs.with_job_mut(job_id, |j| {
                            j.selection.agent = None;
                            j.stage = DispatchStage::ChoosingAgent;
                        });
                        let title = self
                            .pending_jobs
                            .with_job(job_id, |j| j.specification.title.clone())
                            .unwrap_or_default();
                        (
                            format!(
                                "**Feature development: {title}**\n\nChoose a coding agent to implement this feature:"
                            ),
                            develop_agent_components(&id_str),
                        )
                    }
                    Some(DispatchStage::ChoosingEffort) => {
                        let agent = self
                            .pending_jobs
                            .with_job(job_id, |j| j.selection.agent)
                            .flatten();
                        self.pending_jobs.with_job_mut(job_id, |j| {
                            j.selection.model = None;
                            j.stage = DispatchStage::ChoosingModel;
                        });
                        let (title, agent_name) = self
                            .pending_jobs
                            .with_job(job_id, |j| {
                                (
                                    j.specification.title.clone(),
                                    j.selection.agent.map(|a| a.display_name().to_string()),
                                )
                            })
                            .unwrap_or_default();
                        let agent = agent.unwrap_or(CodingAgent::Claude);
                        (
                            format!(
                                "**Feature development: {title}**\n\nAgent: **{}**\nChoose a model:",
                                agent_name.unwrap_or_default()
                            ),
                            develop_model_components(&id_str, agent, &self.catalog),
                        )
                    }
                    Some(DispatchStage::Confirming) => {
                        let agent_opt = self
                            .pending_jobs
                            .with_job(job_id, |j| j.selection.agent)
                            .flatten();
                        let model_opt = self
                            .pending_jobs
                            .with_job(job_id, |j| j.selection.model.clone())
                            .flatten();
                        self.pending_jobs.with_job_mut(job_id, |j| {
                            j.selection.effort = None;
                            j.stage = DispatchStage::ChoosingEffort;
                        });
                        let title = self
                            .pending_jobs
                            .with_job(job_id, |j| j.specification.title.clone())
                            .unwrap_or_default();
                        let agent = agent_opt.unwrap_or(CodingAgent::Claude);
                        let model = model_opt.unwrap_or_default();
                        (
                            format!(
                                "**Feature development: {title}**\n\nAgent: **{}**\nModel: `{model}`\nChoose effort level:",
                                agent.display_name()
                            ),
                            develop_effort_components(&id_str, agent, &model, &self.catalog),
                        )
                    }
                    _ => return,
                };
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content(content)
                                .components(components),
                        ),
                    )
                    .await;
            }
            "cancel" => {
                self.pending_jobs.cancel(job_id);
                let _ = component
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::UpdateMessage(
                            CreateInteractionResponseMessage::new()
                                .content("❌ Development job cancelled.")
                                .components(vec![]),
                        ),
                    )
                    .await;
            }
            _ => {}
        }
    }

    async fn handle_pagination_component(
        &self,
        ctx: &Context,
        component: &serenity::all::ComponentInteraction,
    ) {
        let Some(rest) = component.data.custom_id.strip_prefix(PAGINATION_PREFIX) else {
            return;
        };
        let Some((token, page)) = rest.rsplit_once(':') else {
            return;
        };
        let Ok(page) = page.parse::<usize>() else {
            return;
        };
        let response = self
            .paginated
            .lock()
            .await
            .get(token)
            .map(|response| (response.owner_id, response.pages.clone()));
        let Some((owner_id, pages)) = response else {
            let _ = component
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("This paginated response has expired.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        };
        if owner_id != component.user.id.get() || page >= pages.len() {
            let _ = component
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Only the response author can use these buttons.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }
        let response = CreateInteractionResponse::UpdateMessage(
            CreateInteractionResponseMessage::new()
                .embed(pagination_embed(&pages, page))
                .components(pagination_components(token, page, pages.len())),
        );
        let _ = component.create_response(&ctx.http, response).await;
    }
}

// ── develop flow component builders ──────────────────────────────────────────

fn develop_approval_components(job_id: &str) -> Vec<CreateActionRow> {
    vec![CreateActionRow::Buttons(vec![
        CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:approve"))
            .label("Start work")
            .style(ButtonStyle::Success),
        CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:configure"))
            .label("Change configuration")
            .style(ButtonStyle::Secondary),
        CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:reject"))
            .label("Reject")
            .style(ButtonStyle::Danger),
    ])]
}

fn develop_agent_components(job_id: &str) -> Vec<CreateActionRow> {
    let options = vec![
        CreateSelectMenuOption::new("Codex", "codex"),
        CreateSelectMenuOption::new("Claude Code", "claude"),
        CreateSelectMenuOption::new("OpenCode (NVIDIA)", "opencode"),
    ];
    vec![
        CreateActionRow::SelectMenu(
            CreateSelectMenu::new(
                format!("{DEVELOP_PREFIX}{job_id}:agent"),
                CreateSelectMenuKind::String { options },
            )
            .placeholder("Select coding agent"),
        ),
        CreateActionRow::Buttons(vec![CreateButton::new(format!(
            "{DEVELOP_PREFIX}{job_id}:cancel"
        ))
        .label("Cancel")
        .style(ButtonStyle::Danger)]),
    ]
}

fn develop_model_components(
    job_id: &str,
    agent: CodingAgent,
    catalog: &AgentCatalog,
) -> Vec<CreateActionRow> {
    let models = catalog.models_for(agent);
    let options: Vec<CreateSelectMenuOption> = models
        .iter()
        .map(|m| {
            let mut opt = CreateSelectMenuOption::new(&m.display_name, &m.id);
            if let Some(desc) = &m.description {
                opt = opt.description(desc.chars().take(100).collect::<String>());
            }
            opt
        })
        .collect();
    vec![
        CreateActionRow::SelectMenu(
            CreateSelectMenu::new(
                format!("{DEVELOP_PREFIX}{job_id}:model"),
                CreateSelectMenuKind::String { options },
            )
            .placeholder("Select model"),
        ),
        CreateActionRow::Buttons(vec![
            CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:back"))
                .label("← Back")
                .style(ButtonStyle::Secondary),
            CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:cancel"))
                .label("Cancel")
                .style(ButtonStyle::Danger),
        ]),
    ]
}

fn develop_effort_components(
    job_id: &str,
    agent: CodingAgent,
    model: &str,
    catalog: &AgentCatalog,
) -> Vec<CreateActionRow> {
    let efforts = catalog.efforts_for(agent, model).unwrap_or(&[]);
    let options: Vec<CreateSelectMenuOption> = efforts
        .iter()
        .map(|e| {
            let mut opt = CreateSelectMenuOption::new(&e.display_name, &e.id);
            if let Some(desc) = &e.description {
                opt = opt.description(desc.chars().take(100).collect::<String>());
            }
            opt
        })
        .collect();
    vec![
        CreateActionRow::SelectMenu(
            CreateSelectMenu::new(
                format!("{DEVELOP_PREFIX}{job_id}:effort"),
                CreateSelectMenuKind::String { options },
            )
            .placeholder("Select effort level"),
        ),
        CreateActionRow::Buttons(vec![
            CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:back"))
                .label("← Back")
                .style(ButtonStyle::Secondary),
            CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:cancel"))
                .label("Cancel")
                .style(ButtonStyle::Danger),
        ]),
    ]
}

fn develop_confirm_components(job_id: &str) -> Vec<CreateActionRow> {
    vec![CreateActionRow::Buttons(vec![
        CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:confirm"))
            .label("Dispatch")
            .style(ButtonStyle::Success),
        CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:back"))
            .label("← Change Effort")
            .style(ButtonStyle::Secondary),
        CreateButton::new(format!("{DEVELOP_PREFIX}{job_id}:cancel"))
            .label("Cancel")
            .style(ButtonStyle::Danger),
    ])]
}

fn split_command(content: &str) -> (String, String) {
    match content.split_once('\n') {
        Some((first, rest)) => (first.trim().to_string(), rest.trim().to_string()),
        None => (content.trim().to_string(), String::new()),
    }
}

async fn send_final_message(
    ctx: &Context,
    msg: &Message,
    text: &str,
    paginate: bool,
    owner_id: u64,
    store: &Mutex<HashMap<String, PaginatedResponse>>,
    progress: Option<&Message>,
) {
    if !paginate {
        let chunks = split_text(text, MAX_MESSAGE_LENGTH);
        if let (Some(progress), Some(first)) = (progress, chunks.first()) {
            if progress
                .channel_id
                .edit_message(
                    &ctx.http,
                    progress.id,
                    EditMessage::new()
                        .content(first)
                        .allowed_mentions(CreateAllowedMentions::new()),
                )
                .await
                .is_ok()
            {
                for chunk in chunks.iter().skip(1) {
                    let _ = msg.channel_id.say(&ctx.http, chunk).await;
                }
                return;
            }
        }
        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                let _ = reply_no_ping(ctx, msg, chunk).await;
            } else {
                let _ = msg.channel_id.say(&ctx.http, chunk).await;
            }
        }
        return;
    }

    if let Some(progress) = progress {
        let _ = progress.delete(&ctx.http).await;
    }
    let pages = split_text(text, EMBED_DESCRIPTION_LIMIT);
    let token = uuid::Uuid::new_v4().simple().to_string();
    store.lock().await.insert(
        token.clone(),
        PaginatedResponse {
            owner_id,
            pages: pages.clone(),
        },
    );
    let builder = CreateMessage::new()
        .embed(pagination_embed(&pages, 0))
        .components(pagination_components(&token, 0, pages.len()))
        .reference_message(msg)
        .allowed_mentions(CreateAllowedMentions::new());
    let _ = msg.channel_id.send_message(&ctx.http, builder).await;
}

fn pagination_embed(pages: &[String], page: usize) -> CreateEmbed {
    CreateEmbed::new()
        .description(&pages[page])
        .footer(serenity::all::CreateEmbedFooter::new(format!(
            "Page {} of {}",
            page + 1,
            pages.len()
        )))
}

fn pagination_components(token: &str, page: usize, page_count: usize) -> Vec<CreateActionRow> {
    vec![CreateActionRow::Buttons(vec![
        CreateButton::new(format!(
            "{PAGINATION_PREFIX}{token}:{}",
            page.saturating_sub(1)
        ))
        .label("←")
        .style(ButtonStyle::Secondary)
        .disabled(page == 0),
        CreateButton::new(format!("{PAGINATION_PREFIX}{token}:{}", page + 1))
            .label("→")
            .style(ButtonStyle::Secondary)
            .disabled(page + 1 >= page_count),
    ])]
}

async fn extract_media(msg: &Message) -> Vec<MediaData> {
    let mut media = Vec::new();
    for att in &msg.attachments {
        let Some(media_type) = media_type(&att.filename) else {
            continue;
        };
        if let Ok(resp) = reqwest::get(&att.url).await {
            if let Ok(bytes) = resp.bytes().await {
                use base64::Engine;
                media.push(MediaData {
                    media_type: media_type.to_string(),
                    data: base64::engine::general_purpose::STANDARD.encode(&bytes),
                });
            }
        }
    }
    media
}

fn media_type(filename: &str) -> Option<&'static str> {
    match filename
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Some("image/png"),
        Some("jpg") | Some("jpeg") => Some("image/jpeg"),
        Some("gif") => Some("image/gif"),
        Some("webp") => Some("image/webp"),
        Some("mp3") => Some("audio/mpeg"),
        Some("wav") => Some("audio/wav"),
        Some("flac") => Some("audio/flac"),
        Some("mp4") => Some("video/mp4"),
        Some("mov") => Some("video/quicktime"),
        Some("webm") => Some("video/webm"),
        Some("mkv") => Some("video/x-matroska"),
        Some("avi") => Some("video/x-msvideo"),
        Some("m4v") => Some("video/x-m4v"),
        _ => None,
    }
}

#[cfg(test)]
mod media_tests {
    use super::media_type;

    #[test]
    fn recognizes_supported_media_extensions() {
        assert_eq!(media_type("PHOTO.PNG"), Some("image/png"));
        assert_eq!(media_type("recording.mp3"), Some("audio/mpeg"));
        assert_eq!(media_type("clip.mp4"), Some("video/mp4"));
        assert_eq!(media_type("document.pdf"), None);
    }
}

fn message_has_supported_media(msg: &Message) -> bool {
    msg.attachments
        .iter()
        .any(|attachment| media_type(&attachment.filename).is_some())
}

fn referenced_message_context(msg: &Message) -> Option<String> {
    let text = msg.content.trim();
    let urls: Vec<&str> = URL.find_iter(text).map(|m| m.as_str()).collect();
    let has_media = message_has_supported_media(msg);
    if text.is_empty() && !has_media {
        return None;
    }

    let mut context = String::from("[Message being replied to]\n");
    if !text.is_empty() {
        context.push_str(text);
    }
    if !urls.is_empty() {
        context.push_str(
            "\n\nThe message above contains URL(s). Use the web fetch tool on these URL(s) before answering: ",
        );
        context.push_str(&urls.join(", "));
    }
    if has_media {
        context.push_str("\n\n[The message above also contains media attachment(s) for analysis.]");
    }
    context.push_str("\n[End message being replied to]");
    Some(context)
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Run the bot: build the agent, register the handler, and connect to Discord.
pub async fn run() -> anyhow::Result<()> {
    let token = std::env::var("DISCORD_BOT_TOKEN")
        .map_err(|_| anyhow::anyhow!("DISCORD_BOT_TOKEN is not set"))?;
    let discord = Arc::new(DiscordBridge::default());
    let agent = Arc::new(Agent::from_env(discord.clone()).await);
    let bot = HouseBot::new(agent, discord);

    let intents = GatewayIntents::non_privileged() | GatewayIntents::MESSAGE_CONTENT;
    let mut client = Client::builder(&token, intents).event_handler(bot).await?;
    tracing::info!("Agent and MCP servers ready");
    client.start().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{ProfileTag, UserProfile};
    use serde_json::json;
    use tempfile::TempDir;

    // ── format_lua_reply ──
    #[test]
    fn lua_reply_is_fenced() {
        assert_eq!(format_lua_reply("hello"), "```\nhello\n```");
    }

    #[test]
    fn lua_reply_escapes_nested_fences() {
        let reply = format_lua_reply("a ``` b");
        assert_eq!(reply.matches("```").count(), 2);
    }

    #[test]
    fn lua_reply_fits_discord_limit() {
        let reply = format_lua_reply(&"x".repeat(5000));
        assert!(reply.chars().count() <= MAX_MESSAGE_LENGTH);
        assert!(reply.starts_with("```\n"));
        assert!(reply.ends_with("\n```"));
        assert!(reply.contains('…'));
    }

    #[test]
    fn global_history_combines_profile_and_channel_context() {
        let profile = UserProfile {
            nickname: "Ali".to_string(),
            tags: vec![ProfileTag::WebResearch],
            ..Default::default()
        };
        let history = vec![
            json!({
                "role": "user",
                "content": "Find the release notes",
                "discord_context": {
                    "channel_id": 42,
                    "timestamp": "2026-07-14T20:15:00Z"
                }
            }),
            json!({"role": "assistant", "content": "Here they are"}),
        ];

        let rendered = render_history(&profile, &history);
        assert!(rendered.contains("History for Ali"));
        assert!(rendered.contains("all servers and channels"));
        assert!(rendered.contains("Profile interests: web research"));
        assert!(rendered.contains("[user in <#42> on 2026-07-14]"));
        assert!(
            rendered.find("Find the release notes").unwrap()
                < rendered.find("Here they are").unwrap()
        );
    }

    #[test]
    fn global_history_empty_state_keeps_profile_identity() {
        let profile = UserProfile {
            display_name: "Alice".to_string(),
            ..Default::default()
        };
        let rendered = render_history(&profile, &[]);
        assert!(rendered.contains("History for Alice"));
        assert!(rendered.contains("No conversation history yet."));
    }

    // ── split_text ──
    #[test]
    fn split_short_text_single_chunk() {
        assert_eq!(split_text("hello", 2000), vec!["hello"]);
    }

    #[test]
    fn split_exact_limit_not_split() {
        let text = "a".repeat(2000);
        assert_eq!(split_text(&text, 2000), vec![text.clone()]);
    }

    #[test]
    fn split_over_limit_on_newline() {
        let text = format!("{}\n{}", "a".repeat(1900), "b".repeat(200));
        let chunks = split_text(&text, 2000);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| c.chars().count() <= 2000));
        assert_eq!(chunks.concat(), text.replacen('\n', "", 1));
    }

    #[test]
    fn split_over_limit_no_newline() {
        let text = "x".repeat(2500);
        let chunks = split_text(&text, 2000);
        assert_eq!(chunks, vec!["x".repeat(2000), "x".repeat(500)]);
    }

    #[test]
    fn split_multiple_chunks() {
        let text = vec!["a".repeat(1999); 3].join("\n");
        let chunks = split_text(&text, 2000);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.chars().count() <= 2000));
    }

    #[test]
    fn split_empty_string() {
        assert_eq!(split_text("", 2000), vec![""]);
    }

    #[test]
    fn split_custom_limit() {
        let chunks = split_text("hello\nworld", 6);
        assert_eq!(chunks, vec!["hello", "world"]);
    }

    // ── tool_hint ──
    #[test]
    fn hint_run_skill_with_name_and_input() {
        let h = tool_hint(
            "run_skill",
            &json!({"name": "summarize", "input": "some text"}),
        );
        assert!(h.contains("summarize"));
        assert!(h.contains("some text"));
    }

    #[test]
    fn hint_run_skill_no_name() {
        assert_eq!(tool_hint("run_skill", &json!({"input": "some text"})), "");
    }

    #[test]
    fn hint_falls_back_to_query() {
        assert!(tool_hint("web_search", &json!({"query": "latest news"})).contains("latest news"));
    }

    #[test]
    fn hint_falls_back_to_task() {
        assert!(
            tool_hint("some_tool", &json!({"task": "write a script"})).contains("write a script")
        );
    }

    #[test]
    fn hint_long_value_truncated() {
        let h = tool_hint("some_tool", &json!({"task": "x".repeat(200)}));
        assert!(h.chars().count() <= 85);
    }

    #[test]
    fn hint_unknown_tool_no_known_key() {
        assert_eq!(tool_hint("some_tool", &json!({"foo": "bar"})), "");
    }

    #[test]
    fn hint_multiline_flattened() {
        let h = tool_hint("some_tool", &json!({"task": "line1\nline2"}));
        assert!(!h.contains('\n'));
    }

    #[test]
    fn tool_summary_lists_tools_in_call_order() {
        let summary = append_tool_summary("answer", &["web_search".into(), "translate".into()]);
        assert!(summary.ends_with("🛠️ **Tools used:** `web_search`, `translate`"));
    }

    #[test]
    fn tool_summary_shows_none_when_no_tools_were_called() {
        assert!(append_tool_summary("answer", &[]).ends_with("🛠️ **Tools used:** none"));
    }

    // ── extract_code_files ──
    #[test]
    fn code_short_block_not_extracted() {
        let text = "Here:\n```python\nprint('hi')\n```";
        let (modified, files) = extract_code_files(text);
        assert!(files.is_empty());
        assert!(modified.contains("```"));
    }

    #[test]
    fn code_large_block_extracted() {
        let code = "x = 1\n".repeat(200);
        let text = format!("Here:\n```python\n{code}```");
        let (modified, files) = extract_code_files(&text);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "script_1.py");
        assert_eq!(files[0].1, code.as_bytes());
        assert!(!modified.contains("```"));
        assert!(modified.contains("script_1.py"));
    }

    #[test]
    fn code_extension_from_language() {
        let code = "echo hi\n".repeat(150);
        let (_, files) = extract_code_files(&format!("```bash\n{code}```"));
        assert!(files[0].0.ends_with(".sh"));
    }

    #[test]
    fn code_unknown_language_txt() {
        let code = "blah\n".repeat(200);
        let (_, files) = extract_code_files(&format!("```brainfuck\n{code}```"));
        assert!(files[0].0.ends_with(".txt"));
    }

    #[test]
    fn code_unclosed_block_still_extracted() {
        let code = "x = 1\n".repeat(200);
        let (modified, files) = extract_code_files(&format!("```python\n{code}"));
        assert_eq!(files.len(), 1);
        assert!(modified.contains("script_1.py"));
    }

    #[test]
    fn code_multiple_blocks_numbered() {
        let code = "x = 1\n".repeat(200);
        let (_, files) = extract_code_files(&format!("```python\n{code}```\n```bash\n{code}```"));
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].0, "script_1.py");
        assert_eq!(files[1].0, "script_2.sh");
    }

    #[test]
    fn code_mixed_small_and_large() {
        let small = "print('hi')\n";
        let large = "x = 1\n".repeat(200);
        let (modified, files) =
            extract_code_files(&format!("```python\n{small}```\n```python\n{large}```"));
        assert_eq!(files.len(), 1);
        assert!(modified.contains("script_1.py"));
        assert!(modified.contains("```python"));
    }

    // ── redaction ──
    #[test]
    fn redact_known_secret() {
        let r = SecretRedactor::from_vars([(
            "MY_SECRET_TOKEN".into(),
            "super-secret-token-abc123xyz".into(),
        )]);
        let out = r.redact("The token is super-secret-token-abc123xyz");
        assert!(!out.contains("super-secret-token-abc123xyz"));
        assert!(out.contains("[REDACTED]"));
    }

    #[test]
    fn redact_non_secret_env_not_redacted() {
        let r = SecretRedactor::from_vars([("MY_NAME".into(), "alice-longenough".into())]);
        assert_eq!(r.redact("hello alice-longenough"), "hello alice-longenough");
    }

    #[test]
    fn redact_short_value_not_redacted() {
        let r = SecretRedactor::from_vars([("MY_TOKEN".into(), "abc".into())]);
        assert_eq!(r.redact("abc"), "abc");
    }

    #[test]
    fn redact_multiple_secrets() {
        let r = SecretRedactor::from_vars([
            ("BOT_TOKEN".into(), "discord-token-xyz987".into()),
            ("JELLYFIN_API_KEY".into(), "jellyfin-api-key-456def".into()),
        ]);
        let out = r.redact("token=discord-token-xyz987 key=jellyfin-api-key-456def");
        assert!(!out.contains("discord-token-xyz987"));
        assert!(!out.contains("jellyfin-api-key-456def"));
        assert_eq!(out.matches("[REDACTED]").count(), 2);
    }

    #[test]
    fn redact_text_without_secrets_unchanged() {
        let r = SecretRedactor::from_vars(std::iter::empty());
        assert_eq!(
            r.redact("hello world, no secrets here"),
            "hello world, no secrets here"
        );
    }

    // ── conversation tracker ──
    #[test]
    fn tracker_inactive_when_unknown() {
        let t = ConversationTracker::new(Duration::from_secs(300));
        assert!(!t.is_active(1, 2, Instant::now()));
    }

    #[test]
    fn tracker_active_within_window() {
        let mut t = ConversationTracker::new(Duration::from_secs(300));
        let now = Instant::now();
        t.mark_active(1, 2, now, Duration::from_secs(300));
        assert!(t.is_active(1, 2, now + Duration::from_secs(100)));
    }

    #[test]
    fn tracker_pop_timed_out() {
        let mut t = ConversationTracker::new(Duration::from_secs(300));
        let now = Instant::now();
        t.mark_active(1, 2, now, Duration::from_secs(300));
        assert!(!t.is_active(1, 2, now + Duration::from_secs(400)));
        assert!(t.pop_timed_out(1, 2, now + Duration::from_secs(400)));
        // Now removed.
        assert!(!t.pop_timed_out(1, 2, now + Duration::from_secs(400)));
    }

    // ── commands ──
    #[test]
    fn commit_hash_response_reports_build_sha() {
        assert_eq!(
            commit_hash_response(Some("abcdef1234567890")),
            "Running commit: `abcdef1234567890`"
        );
        assert_eq!(
            commit_hash_response(None),
            "Running commit is unavailable for this build."
        );
    }

    #[test]
    fn proactive_candidate_is_narrow() {
        assert!(is_proactive_candidate("How do I use reminders?"));
        assert!(is_proactive_candidate("Remind me tomorrow"));
        assert!(!is_proactive_candidate("hello everyone"));
    }

    fn stores() -> (TempDir, Skills, Notes, Memory, History) {
        let tmp = TempDir::new().unwrap();
        (
            TempDir::new().unwrap(),
            Skills::new(tmp.path().join("skills.json")),
            Notes::new(tmp.path().join("notes")),
            Memory::new(tmp.path().join("memories")),
            History::new(tmp.path().join("history"), 30),
        )
    }

    #[tokio::test]
    async fn skill_add_and_list() {
        let (_t, skills, _n, _m, _h) = stores();
        let add = skill_command(&skills, "!skill add greeter", "You greet people", 7).await;
        assert!(add.contains("saved"));
        let list = skill_command(&skills, "!skill list", "", 7).await;
        assert!(list.contains("greeter"));
    }

    #[tokio::test]
    async fn skill_invalid_name_rejected() {
        let (_t, skills, _n, _m, _h) = stores();
        let out = skill_command(&skills, "!skill add Bad-Name", "prompt", 1).await;
        assert!(out.contains("lowercase"));
    }

    #[tokio::test]
    async fn skill_delete_missing() {
        let (_t, skills, _n, _m, _h) = stores();
        assert!(skill_command(&skills, "!skill delete nope", "", 1)
            .await
            .contains("not found"));
    }

    #[tokio::test]
    async fn note_save_get_delete() {
        let (_t, _s, notes, _m, _h) = stores();
        assert!(
            note_command(&notes, "!note save shopping", "milk, eggs", 42)
                .await
                .contains("saved")
        );
        assert!(note_command(&notes, "!note get shopping", "", 42)
            .await
            .contains("milk, eggs"));
        assert!(note_command(&notes, "!note delete shopping", "", 42)
            .await
            .contains("deleted"));
        assert!(note_command(&notes, "!note get shopping", "", 42)
            .await
            .contains("not found"));
    }

    #[tokio::test]
    async fn note_list_empty() {
        let (_t, _s, notes, _m, _h) = stores();
        assert!(note_command(&notes, "!note list", "", 1)
            .await
            .contains("no saved notes"));
    }

    #[tokio::test]
    async fn stats_reports_counts() {
        let (_t, skills, notes, memory, history) = stores();
        memory.save(5.to_string(), "some memory").await.unwrap();
        notes.save(5, "a", "x").await.unwrap();
        let out = stats_command(&history, &memory, &notes, &skills, 5, "Alice").await;
        assert!(out.contains("Stats for Alice"));
        assert!(out.contains("Saved notes: 1"));
    }
}
