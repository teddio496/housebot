//! The agentic loop: builds prompts, streams completions from the LLM, dispatches tool
//! calls (built-in tools + MCP servers), and persists per-user history and memory.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Local, Utc};
use serde_json::{json, Value};

use crate::channel_log::ChannelLog;
use crate::coding_agent::pending::PendingJobStore;
use crate::config;
use crate::discord_bridge::DiscordBridge;
use crate::github_issues::GitHubIssueReporter;
use crate::history::History;
use crate::llm::{ChatClient, OpenAiClient, TextSink, ThinkingMode, TokenUsage};
use crate::mcp::McpServer;
use crate::memory::Memory;
use crate::profile::ProfileStore;
use crate::rate_limit::RateLimiter;
use crate::reminders::Reminders;
use crate::skills::{Skill, Skills};
use crate::tools;
use crate::tools::common_crawl::CommonCrawl;
use crate::tools::searxng::SearxNg;
use crate::tools::web_fetch::WebFetch;

/// An inbound media attachment, base64-encoded for the multimodal API.
#[derive(Debug, Clone)]
pub struct MediaData {
    pub media_type: String,
    pub data: String,
}

/// One user turn to run through the agent.
#[derive(Debug, Clone, Copy)]
pub struct AgentRequest<'a> {
    pub user_id: &'a str,
    pub username: &'a str,
    pub text: &'a str,
    pub media: &'a [MediaData],
    /// Optional personality/tone override injected into the system prompt.
    pub personality: Option<&'a str>,
    /// Reasoning budget for this user's requests.
    pub thinking: ThinkingMode,
    /// Discord channel ID (0 if unknown). Used by the `prepare_feature_development` tool.
    pub channel_id: u64,
    /// Whether deep memory (update_memory tool + auto-summary) is enabled for this user.
    pub deep_memory_enabled: bool,
    /// User's display name from their profile (for personalized greetings).
    pub display_name: &'a str,
    /// User's guild nickname from their profile (empty if none).
    pub nickname: &'a str,
    /// User's Discord avatar URL from their persisted profile (empty if none).
    pub avatar_url: &'a str,
    pub profile_tags: &'a str,
    pub quick_actions: &'a str,
    pub guild_id: Option<u64>,
    pub proactive: bool,
    pub record_profile_usage: bool,
}

impl<'a> AgentRequest<'a> {
    /// A plain text request with default settings (used by tests and headless callers).
    pub fn text(user_id: &'a str, username: &'a str, text: &'a str) -> Self {
        Self {
            user_id,
            username,
            text,
            media: &[],
            personality: None,
            thinking: ThinkingMode::default(),
            channel_id: 0,
            deep_memory_enabled: true,
            display_name: username,
            nickname: "",
            avatar_url: "",
            profile_tags: "",
            quick_actions: "",
            guild_id: None,
            proactive: false,
            record_profile_usage: true,
        }
    }
}

/// Structured bot-control action extracted from a tool call, carried alongside text.
#[derive(Debug, Clone)]
pub enum AgentControlAction {
    /// Owner immediate dispatch is ready.
    OwnerDispatchReady { job_id: uuid::Uuid },
    /// Owner wants to configure interactively.
    OwnerConfigurationRequired { job_id: uuid::Uuid },
    /// Non-owner request created; owner must approve.
    OwnerApprovalRequired { job_id: uuid::Uuid },
}

/// The outcome of one `Agent::run`.
#[derive(Debug, Clone, Default)]
pub struct AgentResult {
    pub text: String,
    pub session_notice: Option<String>,
    pub tools_called: Vec<String>,
    /// Set when a `prepare_feature_development` tool call produces a structured outcome.
    pub control_action: Option<AgentControlAction>,
}

#[derive(Debug, Clone, Copy)]
pub struct SessionInfo {
    pub context_tokens: usize,
    pub context_window_tokens: usize,
    pub messages: usize,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}

/// Per-request callbacks used to surface progress into the chat surface.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Cumulative assistant text as it streams in.
    async fn on_text_stream(&self, _partial: &str) {}
    /// A tool is about to run.
    async fn on_tool_called(&self, _tool: &str, _args: &Value) {}
    /// A progress update from a long-running operation.
    async fn on_progress(&self, _line: &str) {}
}

/// No-op hooks (used in tests and headless contexts).
pub struct NoHooks;
#[async_trait]
impl AgentHooks for NoHooks {}

struct TextStreamAdapter<'a>(&'a dyn AgentHooks);
#[async_trait]
impl TextSink for TextStreamAdapter<'_> {
    async fn push(&self, partial: &str) {
        self.0.on_text_stream(partial).await;
    }
}

/// Result of dispatching a single tool call.
enum ToolOutcome {
    Text(String),
    /// A development-flow tool call that also carries a control action.
    DevelopmentAction {
        text: String,
        action: AgentControlAction,
    },
}

/// The agent: LLM client, storage, tools, and connected MCP servers.
pub struct Agent {
    client: Arc<dyn ChatClient>,
    model: String,
    context_window_tokens: usize,
    history: History,
    memory: Memory,
    profile_store: ProfileStore,
    skills: Skills,
    reminders: Reminders,
    reporter: Arc<GitHubIssueReporter>,
    rate_limiter: RateLimiter,
    feature_edit_limiter: RateLimiter,
    /// Non-owner per-user development request limiter.
    non_owner_dev_limiter: RateLimiter,
    /// Owner safety limiter — consumed only at actual GitHub dispatch (reserved for future use).
    #[allow(dead_code)]
    owner_dispatch_limiter: RateLimiter,
    pending_jobs: Arc<PendingJobStore>,
    searxng: SearxNg,
    web_fetch: WebFetch,
    common_crawl: CommonCrawl,
    mcp_servers: Vec<McpServer>,
    session_stats: tokio::sync::Mutex<HashMap<String, SessionStats>>,
    discord: Arc<DiscordBridge>,
    channel_log: ChannelLog,
}

#[derive(Debug, Clone, Copy, Default)]
struct SessionStats {
    requests: u64,
    context_tokens: u64,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
}

impl Agent {
    /// Build an agent from environment configuration and start MCP servers.
    pub async fn from_env(discord: Arc<DiscordBridge>) -> Self {
        let client: Arc<dyn ChatClient> = Arc::new(OpenAiClient::new(
            config::env_or("LLM_BASE_URL", "http://server-slop:8080/v1"),
            config::env_or("LLM_API_KEY", "not-required"),
        ));
        let mcp_servers = start_mcp_servers().await;
        let context_window_tokens = client
            .context_window_tokens()
            .await
            .ok()
            .flatten()
            .map(|tokens| tokens as usize)
            .unwrap_or_else(|| config::env_parse("MAX_CONTEXT_TOKENS", 10_000));
        let memory = match Memory::from_env().await {
            Ok(memory) => memory,
            Err(error) => {
                tracing::warn!(%error, "PostgreSQL memory unavailable, falling back to file-based memory");
                Memory::default()
            }
        };
        Self {
            client,
            model: config::env_or("LLM_MODEL", "gemma-4-12b-qat-q4kxl"),
            context_window_tokens,
            history: History::default(),
            memory,
            profile_store: ProfileStore::default(),
            skills: Skills::default(),
            reminders: Reminders::default(),
            reporter: Arc::new(GitHubIssueReporter::default()),
            rate_limiter: tools::feature_request::default_rate_limiter(),
            feature_edit_limiter: tools::edit_feature_request::default_rate_limiter(),
            non_owner_dev_limiter: tools::feature_development::default_rate_limiter(),
            owner_dispatch_limiter: tools::feature_development::owner_dispatch_limiter(),
            pending_jobs: Arc::new(PendingJobStore::default()),
            searxng: SearxNg::from_env(),
            web_fetch: WebFetch::default(),
            common_crawl: CommonCrawl::default(),
            mcp_servers,
            session_stats: tokio::sync::Mutex::new(HashMap::new()),
            discord,
            channel_log: ChannelLog::default(),
        }
    }

    /// Access to the reminders store (the bot's delivery loop needs it).
    pub fn reminders(&self) -> &Reminders {
        &self.reminders
    }

    /// Shared persistent memory store used by the Discord command surface.
    pub fn memory(&self) -> Memory {
        self.memory.clone()
    }

    /// Shared pending-job store; also held by `HouseBot` to drive the Discord component UI.
    pub fn pending_jobs(&self) -> Arc<PendingJobStore> {
        Arc::clone(&self.pending_jobs)
    }

    /// Access to the GitHub issue reporter (used by `HouseBot` for development job dispatch).
    pub fn reporter(&self) -> &GitHubIssueReporter {
        &self.reporter
    }

    /// Web search for the Lua scripting engine — same SearXNG instance and
    /// rate limits as the agent's `web_search` tool.
    pub async fn web_search(&self, query: &str, max_results: usize) -> String {
        self.searxng
            .search(query, max_results.clamp(1, 20), "")
            .await
    }

    /// Search Jellyfin for the Lua scripting engine, via the MCP server's
    /// search tool (matched by name, since the tool set is server-defined).
    pub async fn jellyfin_search(&self, query: &str) -> String {
        let Some(server) = self.mcp_servers.iter().find(|s| s.prefix == "jellyfin") else {
            return "Error: Jellyfin is not available.".to_string();
        };
        let tools = server.list_tools().await;
        let Some(tool) = tools
            .iter()
            .find(|t| t.name == "search")
            .or_else(|| tools.iter().find(|t| t.name.contains("search")))
        else {
            return "Error: the Jellyfin server exposes no search tool.".to_string();
        };
        match server.call_tool(&tool.name, json!({"query": query})).await {
            Ok(text) => text,
            Err(e) => format!("Error: {e}"),
        }
    }

    // ── session lifecycle ────────────────────────────────────────────────────

    /// Clear conversation history and counters without preserving a summary.
    pub async fn reset_session(&self, user_id: &str) {
        self.session_stats.lock().await.remove(user_id);
        let _ = self.history.clear(user_id).await;
    }

    /// Summarize the current conversation, then start a fresh session.
    pub async fn compact_session(&self, user_id: &str, deep_memory_enabled: bool) {
        self.compact_session_with_hooks(user_id, deep_memory_enabled, &NoHooks)
            .await;
    }

    /// Summarize the current conversation, reporting coarse-grained progress to the caller.
    pub async fn compact_session_with_hooks(
        &self,
        user_id: &str,
        deep_memory_enabled: bool,
        hooks: &dyn AgentHooks,
    ) {
        tracing::info!(target: "housebot::agent", user_id, "Compacting session");
        hooks.on_progress("compact:10").await;
        self.session_stats.lock().await.remove(user_id);
        let past = self.history.load(user_id).await;
        if past.is_empty() {
            hooks.on_progress("compact:100:Nothing to compact.").await;
            return;
        }
        if !deep_memory_enabled {
            let _ = self.history.clear(user_id).await;
            hooks
                .on_progress("compact:100:Conversation cleared without saving a memory summary.")
                .await;
            return;
        }
        hooks.on_progress("compact:25").await;
        let user_memory = self.memory.load(user_id).await;
        let convo: String = past
            .iter()
            .filter_map(|m| {
                let role = m.get("role").and_then(|r| r.as_str())?;
                let content = m.get("content").and_then(|c| c.as_str())?;
                Some(format!("{}: {}", role.to_uppercase(), content))
            })
            .collect::<Vec<_>>()
            .join("\n");

        let truncated: String = convo.chars().take(6000).collect();
        let prompt = format!(
            "The following is a conversation that has ended. Write a concise bullet-point summary \
             of the key facts, preferences, and decisions discussed. This will be appended to the \
             user's memory for future reference. Be brief — 3-8 bullets max.\n\nCONVERSATION:\n{truncated}"
        );
        hooks.on_progress("compact:45").await;
        let completion = self
            .client
            .chat_once(
                &self.model,
                &[json!({"role": "user", "content": prompt})],
                512,
            )
            .await
            .unwrap_or_default();
        self.record_usage(user_id, completion.usage).await;
        let summary = completion.content.unwrap_or_default();

        if !summary.trim().is_empty() {
            let now = Local::now().format("%Y-%m-%d %H:%M");
            let mut updated = String::new();
            if !user_memory.trim().is_empty() {
                updated.push_str(user_memory.trim_end());
                updated.push_str("\n\n");
            }
            updated.push_str(&format!("## Conversation summary ({now})\n{summary}"));
            let _ = self.memory.save(user_id, &updated).await;
        }
        hooks.on_progress("compact:80").await;
        let _ = self.history.clear(user_id).await;
        hooks
            .on_progress("compact:100:Conversation compacted.")
            .await;
    }

    pub fn model_info(&self) -> String {
        format!(
            "**Model**\nName: `{}`\nMax context: ~{} tokens",
            self.model, self.context_window_tokens
        )
    }

    pub async fn session_info(&self, user_id: &str) -> SessionInfo {
        let history = self.history.load(user_id).await;
        let context_window_tokens = self
            .client
            .context_window_tokens()
            .await
            .ok()
            .flatten()
            .map(|tokens| tokens as usize)
            .unwrap_or(self.context_window_tokens);
        let stats = self
            .session_stats
            .lock()
            .await
            .get(user_id)
            .copied()
            .unwrap_or_default();
        SessionInfo {
            context_tokens: stats.context_tokens as usize,
            context_window_tokens,
            messages: history.len(),
            requests: stats.requests,
            input_tokens: stats.input_tokens,
            output_tokens: stats.output_tokens,
            cached_tokens: stats.cached_tokens,
        }
    }

    async fn last_context_tokens(&self, user_id: &str) -> u64 {
        self.session_stats
            .lock()
            .await
            .get(user_id)
            .map_or(0, |stats| stats.context_tokens)
    }

    async fn record_usage(&self, user_id: &str, usage: TokenUsage) {
        let mut all = self.session_stats.lock().await;
        let stats = all.entry(user_id.to_string()).or_default();
        stats.requests += 1;
        stats.context_tokens = usage.prompt_tokens + usage.completion_tokens;
        stats.input_tokens += usage.prompt_tokens;
        stats.output_tokens += usage.completion_tokens;
        stats.cached_tokens += usage.prompt_tokens_details.cached_tokens;
    }

    // ── main loop ────────────────────────────────────────────────────────────

    /// Run one user turn to completion, returning the final assistant text.
    pub async fn run(&self, request: AgentRequest<'_>, hooks: &dyn AgentHooks) -> AgentResult {
        let AgentRequest {
            user_id,
            username,
            text,
            media,
            personality,
            thinking,
            channel_id,
            deep_memory_enabled,
            display_name,
            nickname,
            avatar_url,
            profile_tags,
            quick_actions,
            guild_id,
            proactive,
            record_profile_usage,
        } = request;
        let run_started = std::time::Instant::now();
        tracing::info!(
            target: "housebot::agent",
            user_id,
            username,
            thinking = %thinking,
            text_chars = text.chars().count(),
            media = media.len(),
            "Agent run started"
        );
        let mut user_memory = self.memory.load(user_id).await;
        let mut past = self.history.load(user_id).await;
        let mut session_notice = None;
        let new_user_message = build_user_message(text, media);
        let mut history_user_message = new_user_message.clone();
        history_user_message["discord_context"] = json!({
            "guild_id": guild_id,
            "channel_id": channel_id,
            "timestamp": Utc::now().to_rfc3339(),
            "username": username,
            "display_name": display_name,
            "avatar_url": avatar_url,
        });

        let previous_usage = self.last_context_tokens(user_id).await as f64
            / self.context_window_tokens.max(1) as f64;
        if !past.is_empty() && previous_usage >= 0.8 {
            tracing::info!("Context at 80% for {user_id} — auto-compacting session");
            self.compact_session_with_hooks(user_id, deep_memory_enabled, hooks)
                .await;
            past.clear();
            user_memory = self.memory.load(user_id).await;
            session_notice = Some(
                "⚠️ The context window reached 80%, so I compacted the conversation and started a new session. Use /session to check your current context usage."
                    .into(),
            );
        }

        let all_skills = self.skills.load_all().await;
        let system = json!({
            "role": "system",
            "content": build_system_prompt_with_profile(
                username,
                user_id,
                display_name,
                nickname,
                avatar_url,
                &user_memory,
                &all_skills,
                personality,
                deep_memory_enabled,
                profile_tags,
                quick_actions,
            ),
        });
        let mut messages: Vec<Value> = Vec::with_capacity(past.len() + 2);
        messages.push(system);
        messages.extend(past);
        messages.push(new_user_message.clone());

        let tools = self.build_tools(deep_memory_enabled).await;
        let mut turn_messages: Vec<Value> = Vec::new();
        let mut tools_called = Vec::new();

        let mut control_action: Option<AgentControlAction> = None;

        let final_text = 'agent_loop: loop {
            let text_sink = TextStreamAdapter(hooks);
            let completion = match self
                .client
                .chat_stream(&self.model, &messages, &tools, thinking, Some(&text_sink))
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("LLM error: {e}");
                    break "Sorry, something went wrong contacting the model.".to_string();
                }
            };
            let context_tokens =
                completion.usage.prompt_tokens + completion.usage.completion_tokens;
            self.record_usage(user_id, completion.usage).await;
            let usage = context_tokens as f64 / self.context_window_tokens.max(1) as f64;
            if usage >= 0.7 {
                session_notice = Some(if usage >= 0.8 {
                    "⚠️ The context window reached 80% based on the model's reported usage. It will be compacted automatically before the next message. Use /session to check your current context usage.".into()
                } else {
                    format!(
                        "⚠️ The context window is {:.0}% full based on the model's reported usage. It will compact automatically at 80%. Use /session to check your current context usage.",
                        usage * 100.0
                    )
                });
            }

            let mut assistant = json!({ "role": "assistant", "content": completion.content });
            if !completion.tool_calls.is_empty() {
                assistant["tool_calls"] = Value::Array(
                    completion
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            })
                        })
                        .collect(),
                );
            }
            messages.push(assistant.clone());
            turn_messages.push(assistant);

            let is_tool_turn = completion.finish_reason.as_deref() == Some("tool_calls")
                && !completion.tool_calls.is_empty();
            if !is_tool_turn {
                break completion.content.unwrap_or_default();
            }

            for tc in &completion.tool_calls {
                let args: Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                tools_called.push(tc.name.clone());
                hooks.on_tool_called(&tc.name, &args).await;
                let outcome = self
                    .dispatch_tool(&tc.name, &args, user_id, username, channel_id, hooks)
                    .await;
                let content = match outcome {
                    ToolOutcome::Text(ref t) => t.clone(),
                    ToolOutcome::DevelopmentAction {
                        ref text,
                        ref action,
                    } => {
                        control_action = Some(action.clone());
                        text.clone()
                    }
                };
                let tool_msg = json!({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                });
                messages.push(tool_msg.clone());
                turn_messages.push(tool_msg);

                // Search rate limits are not recoverable within this run. Stop the
                // tool loop after the first limited response so the model cannot keep
                // retrying the search and waiting for another rate-limit window.
                if matches!(tc.name.as_str(), "web_search" | "deep_research")
                    && search_rate_limited(&content)
                {
                    break 'agent_loop "Web search is temporarily rate-limited. Please try again in a few minutes.".to_string();
                }
            }
        };

        if let Err(e) = self
            .history
            .append_turn(user_id, history_user_message, turn_messages)
            .await
        {
            tracing::error!("Failed to save history for {user_id}: {e}");
        }

        // Record only direct-turn tool usage in the user's profile. Proactive
        // replies must not learn profile tags from unsolicited messages.
        if record_profile_usage && !proactive && !tools_called.is_empty() {
            let mut profile = self.profile_store.load(user_id).await;
            for tool_name in &tools_called {
                profile.record_tool_use(tool_name);
            }
            let _ = self.profile_store.save(user_id, &profile).await;
        }

        tracing::info!(
            target: "housebot::agent",
            user_id,
            tools_called = tools_called.len(),
            response_chars = final_text.chars().count(),
            elapsed_ms = run_started.elapsed().as_millis() as u64,
            "Agent run finished"
        );
        AgentResult {
            text: if final_text.is_empty() {
                "(no response)".to_string()
            } else {
                final_text
            },
            session_notice,
            tools_called,
            control_action,
        }
    }

    async fn build_tools(&self, deep_memory_enabled: bool) -> Vec<Value> {
        let mut tools = Vec::new();
        for server in &self.mcp_servers {
            for tool in server.list_tools().await {
                tools.push(to_openai_tool(
                    &format!("{}__{}", server.prefix, tool.name),
                    &tool.description,
                    tool.input_schema,
                ));
            }
        }
        let mut defs: Vec<Value> = vec![
            tools::searxng::definition(),
            tools::searxng::deep_research_definition(),
            tools::web_fetch::definition(),
            tools::common_crawl::definition(),
            run_skill_tool(),
            tools::feature_request::definition(),
            tools::edit_feature_request::definition(),
            tools::feature_development::definition(),
            tools::remind::definition(),
            tools::summarize_url::definition(),
            tools::translate::definition(),
            tools::features::definition(),
            search_messages_tool(),
            get_recent_messages_tool(),
            find_discord_users_tool(),
            get_discord_user_tool(),
        ];
        // Conditionally include update_memory based on user's privacy setting.
        if deep_memory_enabled {
            defs.push(update_memory_tool());
        }
        for def in defs {
            let (name, desc, params) = flatten_tool(&def);
            tools.push(to_openai_tool(&name, &desc, params));
        }
        tools
    }

    async fn dispatch_tool(
        &self,
        name: &str,
        args: &Value,
        user_id: &str,
        username: &str,
        channel_id: u64,
        _hooks: &dyn AgentHooks,
    ) -> ToolOutcome {
        let started = std::time::Instant::now();
        let outcome = self
            .dispatch_tool_inner(name, args, user_id, username, channel_id)
            .await;
        let content = match &outcome {
            ToolOutcome::Text(t) => t.as_str(),
            ToolOutcome::DevelopmentAction { text, .. } => text.as_str(),
        };
        tracing::info!(
            target: "housebot::agent",
            user_id,
            tool = name,
            result_chars = content.chars().count(),
            is_error = content.starts_with("Error:"),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Tool call finished"
        );
        outcome
    }

    async fn dispatch_tool_inner(
        &self,
        name: &str,
        args: &Value,
        user_id: &str,
        username: &str,
        channel_id: u64,
    ) -> ToolOutcome {
        match name {
            "web_search" => ToolOutcome::Text(
                self.searxng
                    .search(
                        str_arg(args, "query"),
                        u64_arg(args, "max_results", 10) as usize,
                        str_arg(args, "language"),
                    )
                    .await,
            ),
            "deep_research" => {
                let questions: Vec<String> = args
                    .get("questions")
                    .and_then(Value::as_array)
                    .map(|questions| {
                        questions
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                ToolOutcome::Text(
                    self.searxng
                        .deep_research(
                            str_arg(args, "topic"),
                            &questions,
                            u64_arg(args, "max_results_per_query", 5) as usize,
                            str_arg(args, "language"),
                        )
                        .await,
                )
            }
            "fetch_webpage" => ToolOutcome::Text(
                self.web_fetch
                    .fetch_content(
                        str_arg(args, "url"),
                        u64_arg(args, "start_index", 0) as usize,
                        u64_arg(args, "max_length", 8000) as usize,
                    )
                    .await,
            ),
            "common_crawl__search" => ToolOutcome::Text(
                self.common_crawl
                    .search(
                        str_arg(args, "pattern"),
                        str_arg(args, "crawl"),
                        args.get("match_type")
                            .and_then(Value::as_str)
                            .unwrap_or("exact"),
                        u64_arg(args, "max_results", 10) as usize,
                    )
                    .await,
            ),
            "update_memory" => {
                let new_content = str_arg(args, "memory_content");
                let _ = self.memory.save(user_id, new_content).await;
                ToolOutcome::Text("Memory updated.".to_string())
            }
            "create_feature_request" => ToolOutcome::Text(
                tools::feature_request::create_feature_request(
                    &self.reporter,
                    &self.rate_limiter,
                    str_arg(args, "title"),
                    str_arg(args, "description"),
                    user_id,
                )
                .await,
            ),
            "edit_feature_request" => ToolOutcome::Text(
                tools::edit_feature_request::edit_feature_request(
                    &self.reporter,
                    &self.feature_edit_limiter,
                    u64_arg(args, "issue_number", 0),
                    args.get("title").and_then(Value::as_str),
                    args.get("description").and_then(Value::as_str),
                    user_id,
                )
                .await,
            ),
            "prepare_feature_development" => {
                use crate::coding_agent::pending::{
                    DevelopmentRequester, DiscordMessageRef, PartialAgentSelection,
                };
                use crate::tools::feature_development::{DispatchMode, FeatureDevelopmentOutcome};

                let requirements: Vec<String> = args
                    .get("requirements")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let acceptance_criteria: Vec<String> = args
                    .get("acceptance_criteria")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();

                let owner_id = config::owner_id();
                let requester_user_id: u64 = user_id.parse().unwrap_or(0);
                let interactive = args
                    .get("interactive")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);

                let dispatch_mode = if requester_user_id == owner_id {
                    if interactive {
                        DispatchMode::Interactive
                    } else {
                        DispatchMode::Immediate
                    }
                } else {
                    DispatchMode::RequireOwnerApproval
                };

                let requester = DevelopmentRequester {
                    user_id: requester_user_id,
                    username: username.to_string(),
                    channel_id,
                    guild_id: None,
                    source_message_id: 0,
                };
                let source_message = DiscordMessageRef {
                    channel_id,
                    message_id: 0,
                };

                let defaults = PartialAgentSelection::default();

                let outcome = tools::feature_development::prepare_feature_development(
                    &self.pending_jobs,
                    &self.non_owner_dev_limiter,
                    owner_id,
                    requester,
                    source_message,
                    str_arg(args, "title"),
                    str_arg(args, "objective"),
                    str_arg(args, "context"),
                    requirements,
                    acceptance_criteria,
                    dispatch_mode,
                    &defaults,
                );

                let text = outcome.tool_response();
                let action = match &outcome {
                    FeatureDevelopmentOutcome::OwnerDispatchReady { job_id } => {
                        Some(AgentControlAction::OwnerDispatchReady { job_id: *job_id })
                    }
                    FeatureDevelopmentOutcome::OwnerConfigurationRequired { job_id } => {
                        Some(AgentControlAction::OwnerConfigurationRequired { job_id: *job_id })
                    }
                    FeatureDevelopmentOutcome::OwnerApprovalRequired { job_id } => {
                        Some(AgentControlAction::OwnerApprovalRequired { job_id: *job_id })
                    }
                    FeatureDevelopmentOutcome::Rejected { .. } => None,
                };
                if let Some(action) = action {
                    ToolOutcome::DevelopmentAction { text, action }
                } else {
                    ToolOutcome::Text(text)
                }
            }
            "set_reminder" => {
                let delay = args
                    .get("delay_minutes")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                ToolOutcome::Text(
                    tools::remind::create_reminder(
                        &self.reminders,
                        user_id,
                        str_arg(args, "message"),
                        delay,
                    )
                    .await,
                )
            }
            "summarize_url" => ToolOutcome::Text(
                tools::summarize_url::fetch_and_summarize(
                    &*self.client,
                    &self.model,
                    str_arg(args, "url"),
                )
                .await,
            ),
            "translate" => ToolOutcome::Text(
                tools::translate::translate_text(
                    &*self.client,
                    &self.model,
                    str_arg(args, "text"),
                    str_arg(args, "target_language"),
                )
                .await,
            ),
            "run_skill" => {
                let skill_name = str_arg(args, "name");
                let input = str_arg(args, "input");
                match self.skills.get(skill_name).await {
                    None => ToolOutcome::Text(format!("Error: Skill '{skill_name}' not found.")),
                    Some(skill) => {
                        let msgs = vec![
                            json!({"role": "system", "content": skill.prompt}),
                            json!({"role": "user", "content": input}),
                        ];
                        let completion = self
                            .client
                            .chat_once(&self.model, &msgs, 4096)
                            .await
                            .unwrap_or_default();
                        ToolOutcome::Text(completion.content.unwrap_or_default())
                    }
                }
            }
            "get_bot_features" => ToolOutcome::Text(tools::features::features_text().to_string()),
            "search_messages" => {
                let query = str_arg(args, "query");
                let max_results = u64_arg(args, "max_results", 10).clamp(1, 20) as usize;
                let target_channel = args
                    .get("channel_id")
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(channel_id);
                ToolOutcome::Text(
                    match self
                        .channel_log
                        .search(target_channel, query, max_results)
                        .await
                    {
                        Err(e) => format!("Error: {e}"),
                        Ok(msgs) if msgs.is_empty() => "No matching messages found.".to_string(),
                        Ok(msgs) => msgs
                            .iter()
                            .map(|m| {
                                let author = m.nick.as_deref().unwrap_or(&m.username);
                                format!("[{}] {}: {}", m.ts, author, m.content)
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    },
                )
            }
            "get_recent_messages" => {
                let minutes = u64_arg(args, "minutes", 30).clamp(1, 1440) as u32;
                let target_channel = args
                    .get("channel_id")
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(channel_id);
                ToolOutcome::Text(
                    match self.channel_log.get_recent(target_channel, minutes).await {
                        Err(e) => format!("Error: {e}"),
                        Ok(msgs) if msgs.is_empty() => {
                            format!("No messages found in the last {minutes} minutes.")
                        }
                        Ok(msgs) => msgs
                            .iter()
                            .map(|m| {
                                let author = m.nick.as_deref().unwrap_or(&m.username);
                                format!("[{}] {}: {}", m.ts, author, m.content)
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    },
                )
            }
            "find_discord_users" => {
                let query = str_arg(args, "query");
                let max_results = u64_arg(args, "max_results", 10).clamp(1, 20) as usize;
                let target_channel = args
                    .get("channel_id")
                    .and_then(Value::as_str)
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(channel_id);
                ToolOutcome::Text(
                    match self
                        .channel_log
                        .find_authors(target_channel, query, max_results)
                        .await
                    {
                        Err(error) => format!("Error: {error}"),
                        Ok(authors) if authors.is_empty() => {
                            "No matching Discord users found in this channel's history.".to_string()
                        }
                        Ok(authors) => authors
                            .iter()
                            .map(|author| {
                                let nick = author.nick.as_deref().unwrap_or("(none)");
                                format!(
                                    "Username: {} | Nickname: {} | ID: {}",
                                    author.username, nick, author.user_id
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    },
                )
            }
            "get_discord_user" => {
                let uid: u64 = str_arg(args, "user_id").parse().unwrap_or(0);
                ToolOutcome::Text(if uid == 0 {
                    "Error: invalid user_id.".to_string()
                } else {
                    match self.discord.fetch_user(uid).await {
                        Ok(u) => {
                            let avatar = u.avatar_url.as_deref().unwrap_or("(none)");
                            format!(
                                "Username: {}\nDisplay name: {}\nID: {}\nBot: {}\nAccount created: {}\nAvatar URL: {}",
                                u.username, u.display_name, u.id, u.bot, u.created_at, avatar
                            )
                        }
                        Err(e) => format!("Error: {e}"),
                    }
                })
            }
            _ if name.contains("__") => {
                let (prefix, tool_name) = name.split_once("__").unwrap();
                for server in &self.mcp_servers {
                    if server.prefix == prefix {
                        return match server.call_tool(tool_name, args.clone()).await {
                            Ok(text) => ToolOutcome::Text(text),
                            Err(e) => ToolOutcome::Text(format!("Error: {e}")),
                        };
                    }
                }
                ToolOutcome::Text(format!("Unknown tool: {name}"))
            }
            _ => ToolOutcome::Text(format!("Unknown tool: {name}")),
        }
    }
}

// ── MCP server configuration ─────────────────────────────────────────────────

async fn start_mcp_servers() -> Vec<McpServer> {
    let mut servers = Vec::new();
    match (
        std::env::var("JELLYFIN_URL"),
        std::env::var("JELLYFIN_API_KEY"),
    ) {
        (Ok(url), Ok(key)) if !url.is_empty() && !key.is_empty() => {
            if let Some(s) = McpServer::start(
                "jellyfin",
                "jellyfin-mcp",
                &["--read-only".to_string()],
                &[
                    ("JELLYFIN_URL".into(), url),
                    ("JELLYFIN_API_KEY".into(), key),
                ],
            )
            .await
            {
                servers.push(s);
            }
        }
        _ => tracing::warn!("JELLYFIN_URL or JELLYFIN_API_KEY not set — Jellyfin MCP disabled"),
    }
    servers
}

// ── pure helpers ─────────────────────────────────────────────────────────────

fn build_user_message(text: &str, media_data: &[MediaData]) -> Value {
    if media_data.is_empty() {
        return json!({"role": "user", "content": text});
    }
    let mut content: Vec<Value> = media_data
        .iter()
        .map(|media| {
            if media.media_type.starts_with("image/") {
                json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:{};base64,{}", media.media_type, media.data)},
                })
            } else if media.media_type.starts_with("audio/") {
                json!({
                    "type": "input_audio",
                    "input_audio": {"data": media.data},
                })
            } else {
                json!({
                    "type": "input_video",
                    "input_video": {"data": media.data},
                })
            }
        })
        .collect();
    content.push(json!({"type": "text", "text": text}));
    json!({"role": "user", "content": content})
}

/// Build the system prompt for a turn.
#[allow(clippy::too_many_arguments)]
pub fn build_system_prompt(
    username: &str,
    user_id: &str,
    display_name: &str,
    nickname: &str,
    user_memory: &str,
    all_skills: &BTreeMap<String, Skill>,
    personality: Option<&str>,
    deep_memory_enabled: bool,
) -> String {
    build_system_prompt_with_profile(
        username,
        user_id,
        display_name,
        nickname,
        "",
        user_memory,
        all_skills,
        personality,
        deep_memory_enabled,
        "",
        "",
    )
}

#[allow(clippy::too_many_arguments)]
fn build_system_prompt_with_profile(
    username: &str,
    user_id: &str,
    display_name: &str,
    nickname: &str,
    avatar_url: &str,
    user_memory: &str,
    all_skills: &BTreeMap<String, Skill>,
    personality: Option<&str>,
    deep_memory_enabled: bool,
    profile_tags: &str,
    quick_actions: &str,
) -> String {
    let now = Local::now().format("%Y-%m-%d %H:%M");
    let memory_section = if user_memory.trim().is_empty() {
        String::new()
    } else {
        format!("\n\n## Your memory about {username}\n{user_memory}")
    };
    let personality_section = match personality {
        Some(p) if !p.trim().is_empty() => {
            format!("\n\n## Personality / tone for this user\n{}", p.trim())
        }
        _ => String::new(),
    };
    let profile_section = if display_name != username
        || !nickname.is_empty()
        || !avatar_url.is_empty()
        || !profile_tags.is_empty()
        || !quick_actions.is_empty()
    {
        let name_line = if !nickname.is_empty() {
            format!("Display name: {display_name}, Nickname: {nickname}")
        } else {
            format!("Display name: {display_name}")
        };
        let tags_line = if profile_tags.is_empty() {
            String::new()
        } else {
            format!("\nRelevant usage tags: {profile_tags}")
        };
        let avatar_line = if avatar_url.is_empty() {
            String::new()
        } else {
            format!("\nAvatar URL: {avatar_url}")
        };
        let actions_line = if quick_actions.is_empty() {
            String::new()
        } else {
            format!("\nFrequently used actions: {quick_actions}")
        };
        format!(
            "\n\n## User profile\n{name_line}{avatar_line}{tags_line}{actions_line}\n\
             Personalization guidance:\n\
             - If the user greets you, naturally address them by their nickname or display name.\n\
             - If they ask what to do or how you can help, suggest at most one relevant quick action.\n\
             - Use profile tags only to prioritize relevant help; do not announce, expose, or speculate about the profile.\n\
             - Never infer sensitive traits or make unsolicited personal claims from usage patterns."
        )
    } else {
        String::new()
    };
    let memory_guidance = if deep_memory_enabled {
        "Use the saved memory to personalize this conversation naturally, and update it when you learn a durable preference, fact, ongoing project, or correction worth remembering. Never mention the memory store unless the user asks about it."
    } else {
        "Deep memory is disabled for this user. Do NOT call update_memory and do NOT suggest \
         persisting facts. Short-term conversation history within this session still works normally."
    };
    let memory_tool_line = if deep_memory_enabled {
        "- update_memory — Persist important facts about the current user for future conversations. Write the full memory each time.\n"
    } else {
        ""
    };
    let skills_section = if all_skills.is_empty() {
        "\n- run_skill — Execute a custom skill by name. No skills are defined yet; users can add \
         them with `!skill add`."
            .to_string()
    } else {
        let lines: Vec<String> = all_skills
            .values()
            .map(|s| format!("  - **{}**: {}", s.name, s.description_or_name()))
            .collect();
        format!(
            "\n- run_skill — Execute a custom skill by name with an input string. Available skills:\n{}",
            lines.join("\n")
        )
    };
    format!(
        "You are a helpful house assistant bot in a Discord server. You help with media, web \
search, general information, and software development questions.\n\nCurrent date/time: {now}\nCurrent user: {username} \
(ID: {user_id}){profile_section}{memory_section}{personality_section}\n\n## Tools\n\
- web_search — Search the web (SearXNG) for current information.\n\
- deep_research — Run an overview plus 2-5 focused searches and return a deduplicated, cross-referenced source dossier.\n\
- fetch_webpage — Fetch and read the text of a public webpage.\n\
- common_crawl__search — Search historical URL captures in the Common Crawl index.\n\
- jellyfin__* — Query the household Jellyfin media server for movies, shows, music. \
READ ONLY — only call get_* / search_* / list_* methods; never call mutating actions.\n\
{memory_tool_line}\
- create_feature_request — File a GitHub issue for a feature the user wants added to this bot.\n\
- edit_feature_request — Edit a feature request filed by the current user; ownership is verified by the tool.\n\
- prepare_feature_development — Prepare an automated coding-agent development job for the configured bot owner to review and confirm. Only call this when the owner explicitly asks to have a feature automatically implemented by a coding agent. For ordinary feature suggestions, use create_feature_request instead.\n\
- set_reminder — Set a timed reminder; the bot will DM the user when the delay elapses.\n\
- summarize_url — Fetch a public web URL and return a concise summary.\n\
- translate — Translate text to any language using the LLM.\n\
- get_bot_features — Return the full list of this bot's commands and capabilities. \
Call this when a user asks what you can do, what commands exist, or how to use any feature.\n\
- search_messages — Search the current channel's message log by regex pattern. Only matching \
messages are returned, keeping token usage low. Use this when a user asks what was said, who \
mentioned something, or what was discussed. Prefer a targeted pattern over a broad one.\n\
- find_discord_users — Resolve a username or nickname to users seen in the current channel.\n\
- get_discord_user — Look up a Discord user's profile by their user ID (username, display name, \
account creation date, bot status).{skills_section}\n\n\
## Guidelines\n- Be conversational and friendly.\n- Use Jellyfin tools for any media questions \
before guessing.\n- Never infer sensitive traits, identity, or intent from a user's avatar.\n- Use web_search for simple factual or current-events questions. For complex questions requiring multiple perspectives, comparisons, or a comprehensive report, use deep_research and synthesize its dossier with source links. If either search tool returns a rate-limit \
error, stop using search tools for this request and do not retry repeatedly; use \
common_crawl__search for historical URL evidence when appropriate, or explain that the search \
service is temporarily unavailable.\n- You can discuss, explain, review, and advise on software \
development, but you cannot execute code.\n- {memory_guidance}\n- Keep responses concise unless asked for detail.\n- If a user \
requests a feature or improvement to this bot, immediately call create_feature_request with a \
clear title and description, then tell them the issue URL.\n- If a tool returns an error message \
(starts with \"Error:\"), quote it exactly — do not paraphrase or soften it.\n- When the user's \
message exceeds 500 characters, begin your reply with a **TL;DR:** line (one sentence) \
summarizing what they asked.\n"
    )
}

fn update_memory_tool() -> Value {
    json!({
        "name": "update_memory",
        "description": "Update your persistent memory about the current user. Write the complete \
            updated memory content each time, not just the new piece.",
        "input_schema": {
            "type": "object",
            "properties": {"memory_content": {"type": "string", "description": "Full updated memory in markdown format."}},
            "required": ["memory_content"]
        }
    })
}

fn run_skill_tool() -> Value {
    json!({
        "name": "run_skill",
        "description": "Execute a named skill — a custom prompt template saved by users. Pass the \
            skill name and the text input to process.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill name to execute."},
                "input": {"type": "string", "description": "The text input to pass to the skill."}
            },
            "required": ["name", "input"]
        }
    })
}

/// Wrap a tool in the OpenAI function-calling envelope.
pub fn to_openai_tool(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    })
}

/// Convert an internal tool definition into `(name, description, parameters)`.
pub fn flatten_tool(tool_def: &Value) -> (String, String, Value) {
    let name = tool_def
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string();
    let description = tool_def
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();
    let parameters = tool_def
        .get("input_schema")
        .or_else(|| tool_def.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    (name, description, parameters)
}

/// Extract a string argument from tool-call args, defaulting to empty.
fn str_arg<'a>(args: &'a Value, key: &str) -> &'a str {
    args.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Extract an unsigned integer argument from tool-call args.
fn u64_arg(args: &Value, key: &str, default: u64) -> u64 {
    args.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn search_messages_tool() -> Value {
    json!({
        "name": "search_messages",
        "description": "Search Discord channel messages by regex pattern. The pattern is matched \
            against message content, the author's Discord username, AND the author's server \
            nickname or display name. Use this when a user asks what someone said or what was \
            discussed — e.g. to find all messages by 'hexagone', search for '(?i)hexagone' and \
            it will match any message where that name appears as the author or in the text. \
            Supports full Rust regex syntax; case-insensitive patterns ((?i)) are common.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Regex pattern matched against message content, author username, and author nickname/display name."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (1–20, default 10). \
                        Matches are the most recent ones in the log."
                },
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID to search. Omit to search the current channel."
                }
            },
            "required": ["query"]
        }
    })
}

fn get_recent_messages_tool() -> Value {
    json!({
        "name": "get_recent_messages",
        "description": "Return all messages from the current channel posted in the last N minutes, \
            in chronological order. Use this to summarize a recent conversation, catch up on what \
            was discussed, or answer questions like 'what happened in the last 30 minutes'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "minutes": {
                    "type": "integer",
                    "description": "How far back to look, in minutes (1–1440, default 30)."
                },
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID to fetch. Omit to use the current channel."
                }
            },
            "required": []
        }
    })
}

fn get_discord_user_tool() -> Value {
    json!({
        "name": "get_discord_user",
        "description": "Fetch public profile information for a Discord user by their user ID. \
            Returns the username, display name, account creation date, and whether the account \
            is a bot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The Discord user ID (snowflake) to look up."
                }
            },
            "required": ["user_id"]
        }
    })
}

fn find_discord_users_tool() -> Value {
    json!({
        "name": "find_discord_users",
        "description": "Find Discord users previously seen in the current channel by username, nickname, or user ID. Use this before get_discord_user when a person is named but their numeric ID is unknown. Results are limited to the selected channel's message history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Case-insensitive username, nickname, or user ID substring."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of users to return (1–20, default 10)."
                },
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID to search. Omit to use the current channel."
                }
            },
            "required": ["query"]
        }
    })
}

fn search_rate_limited(content: &str) -> bool {
    let content = content.to_ascii_lowercase();
    content.contains("returned http 429")
        || content.contains("too many requests")
        || content.contains("rate limit")
        || content.contains("temporarily blocked")
}

#[cfg(test)]
impl Agent {
    /// Construct an agent wired to a test client and temp-backed stores.
    pub fn for_test(
        client: Arc<dyn ChatClient>,
        history: History,
        memory: Memory,
        profile_store: ProfileStore,
        skills: Skills,
        reminders: Reminders,
    ) -> Self {
        Self {
            client,
            model: "test-model".into(),
            context_window_tokens: 10_000,
            history,
            memory,
            profile_store,
            skills,
            reminders,
            reporter: Arc::new(GitHubIssueReporter::new(
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            )),
            rate_limiter: tools::feature_request::default_rate_limiter(),
            feature_edit_limiter: tools::edit_feature_request::default_rate_limiter(),
            non_owner_dev_limiter: tools::feature_development::default_rate_limiter(),
            owner_dispatch_limiter: tools::feature_development::owner_dispatch_limiter(),
            pending_jobs: Arc::new(PendingJobStore::default()),
            searxng: SearxNg::from_env(),
            web_fetch: WebFetch::default(),
            common_crawl: CommonCrawl::default(),
            mcp_servers: vec![],
            session_stats: tokio::sync::Mutex::new(HashMap::new()),
            discord: Arc::new(DiscordBridge::default()),
            channel_log: ChannelLog::default(),
        }
    }

    pub fn set_max_context_tokens(&mut self, n: usize) {
        self.context_window_tokens = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockChatClient;
    use tempfile::TempDir;

    fn empty_skills() -> BTreeMap<String, Skill> {
        BTreeMap::new()
    }

    #[test]
    fn system_prompt_includes_username_and_id() {
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true);
        assert!(p.contains("Alice"));
        assert!(p.contains("123"));
    }

    #[test]
    fn system_prompt_memory_section_present_when_nonempty() {
        let p = build_system_prompt(
            "Alice",
            "123",
            "Alice",
            "",
            "Likes cats",
            &empty_skills(),
            None,
            true,
        );
        assert!(p.contains("Likes cats"));
        assert!(p.contains("Your memory"));
    }

    #[test]
    fn system_prompt_memory_absent_when_blank() {
        assert!(!build_system_prompt(
            "Alice",
            "123",
            "Alice",
            "",
            "   ",
            &empty_skills(),
            None,
            true
        )
        .contains("Your memory"));
    }

    #[test]
    fn system_prompt_lists_skills() {
        let mut skills = BTreeMap::new();
        skills.insert(
            "greet".into(),
            Skill {
                name: "greet".into(),
                description: Some("Say hello".into()),
                prompt: "..".into(),
                created_by: None,
            },
        );
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &skills, None, true);
        assert!(p.contains("greet"));
        assert!(p.contains("Say hello"));
    }

    #[test]
    fn system_prompt_placeholder_without_skills() {
        assert!(
            build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true)
                .contains("No skills are defined yet")
        );
    }

    #[test]
    fn system_prompt_has_tldr_and_500() {
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true);
        assert!(p.contains("TL;DR"));
        assert!(p.contains("500"));
    }

    #[test]
    fn system_prompt_routes_complex_questions_to_deep_research() {
        let p = build_system_prompt("Alice", "123", "", "", "", &empty_skills(), None, true);
        assert!(p.contains("deep_research"));
        assert!(p.contains("multiple perspectives"));
        assert!(p.contains("source links"));
    }

    #[test]
    fn system_prompt_excludes_code_execution() {
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true);
        assert!(!p.contains("code execution"));
    }

    #[test]
    fn system_prompt_lists_discord_user_tools_once_and_in_order() {
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true);
        let find = p.find("- find_discord_users —").unwrap();
        let get = p.find("- get_discord_user —").unwrap();
        assert!(find < get);
        assert_eq!(p.matches("- get_discord_user —").count(), 1);
    }

    #[test]
    fn system_prompt_includes_profile_section_with_nickname() {
        let p = build_system_prompt(
            "Alice",
            "123",
            "Alice",
            "Ali",
            "",
            &empty_skills(),
            None,
            true,
        );
        assert!(p.contains("User profile"));
        assert!(p.contains("Nickname: Ali"));
    }

    #[test]
    fn system_prompt_skips_profile_section_when_identical() {
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true);
        assert!(!p.contains("User profile"));
    }

    #[test]
    fn system_prompt_includes_usage_profile() {
        let p = build_system_prompt_with_profile(
            "Alice",
            "123",
            "Alice",
            "",
            "",
            "",
            &empty_skills(),
            None,
            true,
            "media, reminders",
            "media (4), reminders (2)",
        );
        assert!(p.contains("Relevant usage tags: media, reminders"));
        assert!(p.contains("Frequently used actions: media (4), reminders (2)"));
        assert!(p.contains("naturally address them by their nickname or display name"));
        assert!(p.contains("suggest at most one relevant quick action"));
        assert!(p.contains("Never infer sensitive traits"));
    }

    #[test]
    fn system_prompt_includes_profile_avatar_with_safety_guidance() {
        let p = build_system_prompt_with_profile(
            "Alice",
            "123",
            "Alice",
            "",
            "https://cdn.discordapp.com/avatars/123/avatar.png",
            "",
            &empty_skills(),
            None,
            true,
            "",
            "",
        );
        assert!(p.contains("Avatar URL: https://cdn.discordapp.com/avatars/123/avatar.png"));
        assert!(
            p.contains("Never infer sensitive traits, identity, or intent from a user's avatar.")
        );
    }

    #[test]
    fn system_prompt_respects_deep_memory_disabled() {
        let p = build_system_prompt(
            "Alice",
            "123",
            "Alice",
            "",
            "",
            &empty_skills(),
            None,
            false,
        );
        assert!(p.contains("Deep memory is disabled"));
        assert!(p.contains("Do NOT call update_memory"));
    }

    #[test]
    fn system_prompt_allows_deep_memory_when_enabled() {
        let p = build_system_prompt("Alice", "123", "Alice", "", "", &empty_skills(), None, true);
        assert!(p.contains("Use the saved memory to personalize this conversation naturally"));
    }

    #[test]
    fn flatten_tool_extracts_fields() {
        let tool = json!({"name": "my_tool", "description": "does stuff", "input_schema": {"type": "object"}});
        let (n, d, p) = flatten_tool(&tool);
        assert_eq!(n, "my_tool");
        assert_eq!(d, "does stuff");
        assert_eq!(p, json!({"type": "object"}));
    }

    #[test]
    fn flatten_tool_falls_back_to_parameters() {
        let tool = json!({"name": "t", "parameters": {"type": "object"}});
        assert_eq!(flatten_tool(&tool).2, json!({"type": "object"}));
    }

    #[test]
    fn to_openai_tool_wraps_in_envelope() {
        let t = to_openai_tool("my_tool", "does stuff", json!({"type": "object"}));
        assert_eq!(t["type"], "function");
        assert_eq!(t["function"]["name"], "my_tool");
        assert_eq!(t["function"]["parameters"], json!({"type": "object"}));
    }

    #[test]
    fn search_rate_limit_errors_are_detected() {
        assert!(search_rate_limited(
            "Error: SearXNG returned HTTP 429 Too Many Requests"
        ));
        assert!(search_rate_limited("SearXNG rate limit reached"));
        assert!(!search_rate_limited(
            "Error: search request failed: timeout"
        ));
    }

    #[test]
    fn build_user_message_plain_text() {
        let m = build_user_message("hi", &[]);
        assert_eq!(m["content"], "hi");
    }

    #[test]
    fn build_user_message_with_image() {
        let imgs = vec![MediaData {
            media_type: "image/png".into(),
            data: "abc".into(),
        }];
        let m = build_user_message("look", &imgs);
        assert_eq!(m["content"][0]["type"], "image_url");
        assert!(m["content"][0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .contains("data:image/png;base64,abc"));
        assert_eq!(m["content"][1]["text"], "look");
    }

    #[test]
    fn build_user_message_with_audio_and_video() {
        let media = vec![
            MediaData {
                media_type: "audio/mpeg".into(),
                data: "audio-bytes".into(),
            },
            MediaData {
                media_type: "video/mp4".into(),
                data: "video-bytes".into(),
            },
        ];
        let message = build_user_message("analyze", &media);
        assert_eq!(message["content"][0]["type"], "input_audio");
        assert_eq!(message["content"][0]["input_audio"]["data"], "audio-bytes");
        assert_eq!(message["content"][1]["type"], "input_video");
        assert_eq!(message["content"][1]["input_video"]["data"], "video-bytes");
    }

    fn test_agent(client: Arc<dyn ChatClient>) -> (TempDir, Agent) {
        let tmp = TempDir::new().unwrap();
        let agent = Agent::for_test(
            client,
            History::new(tmp.path().join("history"), 30),
            Memory::new(tmp.path().join("memories")),
            ProfileStore::new(tmp.path().join("profiles")),
            Skills::new(tmp.path().join("skills.json")),
            Reminders::new(tmp.path().join("reminders.json")),
        );
        (tmp, agent)
    }

    #[tokio::test]
    async fn run_returns_plain_text_completion() {
        let client = Arc::new(MockChatClient::new());
        client.push_text("hello there");
        let (_t, agent) = test_agent(client);
        let result = agent
            .run(AgentRequest::text("u1", "Alice", "hi"), &NoHooks)
            .await;
        assert_eq!(result.text, "hello there");
    }

    #[tokio::test]
    async fn run_persists_history() {
        let client = Arc::new(MockChatClient::new());
        client.push_text("saved reply");
        let (_t, agent) = test_agent(client);
        agent
            .run(AgentRequest::text("u2", "Bob", "remember this"), &NoHooks)
            .await;
        let hist = agent.history.load("u2").await;
        assert_eq!(hist.len(), 2); // user + assistant
        assert_eq!(hist[0]["content"], "remember this");
    }

    #[tokio::test]
    async fn run_dispatches_translate_tool_then_answers() {
        let client = Arc::new(MockChatClient::new().with_once_reply("Bonjour"));
        // First completion asks for a translate tool call; second finishes with text.
        client.push_tool_call(
            "call_1",
            "translate",
            r#"{"text":"Hello","target_language":"French"}"#,
        );
        client.push_text("It means Bonjour.");
        let (_t, agent) = test_agent(client);
        let result = agent
            .run(
                AgentRequest::text("u3", "Cy", "translate Hello to French"),
                &NoHooks,
            )
            .await;
        assert_eq!(result.text, "It means Bonjour.");
        // History should contain the assistant tool-call turn and the tool result.
        let hist = agent.history.load("u3").await;
        assert!(hist
            .iter()
            .any(|m| m["role"] == "tool" && m["content"] == "Bonjour"));
    }

    #[tokio::test]
    async fn run_update_memory_tool_persists() {
        let client = Arc::new(MockChatClient::new());
        client.push_tool_call("c1", "update_memory", r#"{"memory_content":"Likes tea"}"#);
        client.push_text("Noted.");
        let (_t, agent) = test_agent(client);
        agent
            .run(
                AgentRequest::text("u4", "Dee", "remember I like tea"),
                &NoHooks,
            )
            .await;
        assert_eq!(agent.memory.load("u4").await, "Likes tea");
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_returns_error() {
        let client = Arc::new(MockChatClient::new());
        let (_t, agent) = test_agent(client);
        let out = agent
            .dispatch_tool(
                "run_unknown_code_agent",
                &json!({}),
                "u",
                "testuser",
                0,
                &NoHooks,
            )
            .await;
        match out {
            ToolOutcome::Text(t) => assert!(t.contains("Unknown tool")),
            ToolOutcome::DevelopmentAction { text, .. } => {
                panic!("unexpected development action: {text}")
            }
        }
    }

    #[tokio::test]
    async fn context_overflow_triggers_new_session() {
        let client = Arc::new(MockChatClient::new());
        client.push_text_with_usage(
            "ok",
            TokenUsage {
                prompt_tokens: 40,
                completion_tokens: 10,
                ..Default::default()
            },
        );
        client.push_text("ok again");
        let tmp = TempDir::new().unwrap();
        let mut agent = Agent::for_test(
            client,
            History::new(tmp.path().join("history"), 30),
            Memory::new(tmp.path().join("memories")),
            ProfileStore::new(tmp.path().join("profiles")),
            Skills::new(tmp.path().join("skills.json")),
            Reminders::new(tmp.path().join("reminders.json")),
        );
        agent.set_max_context_tokens(50);
        let big = "x".repeat(200);
        agent
            .history
            .save(
                "u5",
                &[
                    json!({"role": "user", "content": big.clone()}),
                    json!({"role": "assistant", "content": "ok"}),
                ],
            )
            .await
            .unwrap();

        agent
            .run(AgentRequest::text("u5", "Ed", "hi again"), &NoHooks)
            .await;
        agent
            .run(AgentRequest::text("u5", "Ed", "one more"), &NoHooks)
            .await;

        // The oversized message must have been summarized away; only the new turn remains.
        let hist = agent.history.load("u5").await;
        assert!(!hist
            .iter()
            .any(|m| m["content"].as_str() == Some(big.as_str())));
        assert_eq!(hist.last().unwrap()["content"], "ok again");
    }

    #[tokio::test]
    async fn compaction_records_summary_token_usage() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        };
        let client = Arc::new(
            MockChatClient::new()
                .with_once_reply("- Likes tea")
                .with_once_usage(usage),
        );
        let (_t, agent) = test_agent(client);
        agent
            .history
            .save(
                "u6",
                &[
                    json!({"role": "user", "content": "I like tea"}),
                    json!({"role": "assistant", "content": "Noted"}),
                ],
            )
            .await
            .unwrap();

        agent.compact_session("u6", true).await;

        let info = agent.session_info("u6").await;
        assert_eq!(info.context_tokens, 150);
        assert_eq!(info.requests, 1);
        assert_eq!(info.input_tokens, 100);
        assert_eq!(info.output_tokens, 50);
    }

    #[tokio::test]
    async fn disabled_memory_compaction_clears_history_without_writing_memory() {
        let client = Arc::new(MockChatClient::new().with_once_reply("should not be called"));
        let (_t, agent) = test_agent(client);
        agent.memory.save("u7", "Keep this memory").await.unwrap();
        agent
            .history
            .save(
                "u7",
                &[
                    json!({"role": "user", "content": "private conversation"}),
                    json!({"role": "assistant", "content": "reply"}),
                ],
            )
            .await
            .unwrap();

        agent.compact_session("u7", false).await;

        assert_eq!(agent.memory.load("u7").await, "Keep this memory");
        assert!(agent.history.load("u7").await.is_empty());
    }

    #[tokio::test]
    async fn history_turn_contains_discord_context_metadata() {
        let client = Arc::new(MockChatClient::new().with_once_reply("ok"));
        let (_t, agent) = test_agent(client);
        let mut request = AgentRequest::text("u8", "alice", "hello");
        request.channel_id = 42;
        request.guild_id = Some(7);
        request.display_name = "Alice";
        request.avatar_url = "https://cdn.discordapp.com/avatars/u8/avatar.png";
        agent.run(request, &NoHooks).await;

        let history = agent.history.load("u8").await;
        assert_eq!(history[0]["discord_context"]["guild_id"], 7);
        assert_eq!(history[0]["discord_context"]["channel_id"], 42);
        assert_eq!(history[0]["discord_context"]["username"], "alice");
        assert_eq!(
            history[0]["discord_context"]["avatar_url"],
            "https://cdn.discordapp.com/avatars/u8/avatar.png"
        );
        assert!(history[0]["discord_context"]["timestamp"].is_string());
    }

    #[tokio::test]
    async fn build_tools_excludes_code_execution() {
        let client = Arc::new(MockChatClient::new());
        let (_t, agent) = test_agent(client);
        let tools = agent.build_tools(true).await;
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(!names.contains(&"code_tool"));
        assert!(names.contains(&"translate"));
        assert!(names.contains(&"update_memory"));
        assert!(names.contains(&"common_crawl__search"));
        assert!(names.contains(&"find_discord_users"));
        assert!(names.contains(&"edit_feature_request"));
        assert!(names.contains(&"deep_research"));
    }
}
