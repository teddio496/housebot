//! Small Discord bot that observes deployment webhooks and offers owner-only deployment controls.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::Deserialize;
use serenity::all::{
    ButtonStyle, Command, CommandDataOptionValue, CommandOptionType, Context, CreateActionRow,
    CreateButton, CreateCommand, CreateCommandOption, CreateEmbed, CreateInteractionResponse,
    CreateInteractionResponseMessage, EditInteractionResponse, EventHandler, GatewayIntents,
    GuildId, Interaction, Message, Ready,
};
use serenity::Client;
use tokio::process::Command as ProcessCommand;
use tokio::sync::{Mutex, RwLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeploymentEvent {
    pub succeeded: bool,
    pub commit: Option<String>,
}

pub fn classify_deployment_text(text: &str) -> Option<bool> {
    let normalized = text.to_ascii_lowercase();
    if normalized.contains("deployment succeeded") || normalized.contains("build succeeded") {
        Some(true)
    } else if normalized.contains("deployment failed") {
        Some(false)
    } else {
        None
    }
}

pub fn deployment_event(message: &Message) -> Option<DeploymentEvent> {
    message.webhook_id?;
    let text = message
        .embeds
        .iter()
        .flat_map(|embed| {
            embed
                .title
                .iter()
                .chain(embed.description.iter())
                .chain(embed.fields.iter().map(|field| &field.value))
        })
        .chain(std::iter::once(&message.content))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let succeeded = classify_deployment_text(&text)?;
    let commit = message
        .embeds
        .iter()
        .flat_map(|embed| &embed.fields)
        .find(|field| field.name.eq_ignore_ascii_case("commit"))
        .map(|field| field.value.trim_matches('`').to_string());
    Some(DeploymentEvent { succeeded, commit })
}

pub fn rollback_allowed(
    owner_id: u64,
    requesting_user: u64,
    channel: u64,
    expected_channel: u64,
) -> bool {
    owner_id != 0 && owner_id == requesting_user && channel == expected_channel
}

#[derive(Clone)]
struct DeploymentBot {
    owner_id: u64,
    channel_id: u64,
    guild_id: Option<u64>,
    last_event: Arc<RwLock<Option<DeploymentEvent>>>,
    previous_image: Arc<RwLock<Option<String>>>,
    deployment_lock: Arc<Mutex<()>>,
    github_repo: String,
    github_branch: String,
    github_token: Option<String>,
    docker_network: String,
}

const HOUSE_CHATBOT_CONTAINER: &str = "house-chatbot";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeploymentStage {
    PullHousebotImage,
    RemovePreviousContainer,
    StartRequestedImage,
    CheckContainerState,
}

impl DeploymentStage {
    pub fn progress_message(self) -> &'static str {
        match self {
            Self::PullHousebotImage => "⬇️ Pulling housebot image…",
            Self::RemovePreviousContainer => "🛑 Removing the previous housebot container…",
            Self::StartRequestedImage => "🚀 Starting the requested housebot image…",
            Self::CheckContainerState => "🩺 Checking container state…",
        }
    }

    fn is_start(self) -> bool {
        self == Self::StartRequestedImage
    }

    fn is_health_check(self) -> bool {
        self == Self::CheckContainerState
    }
}

impl fmt::Display for DeploymentStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PullHousebotImage => "pull_housebot_image",
            Self::RemovePreviousContainer => "remove_previous_container",
            Self::StartRequestedImage => "start_requested_image",
            Self::CheckContainerState => "check_container_state",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeploymentCommand {
    pub stage: DeploymentStage,
    pub args: Vec<String>,
}

impl DeploymentCommand {
    fn new(stage: DeploymentStage, args: Vec<String>) -> Self {
        Self { stage, args }
    }

    fn args(&self) -> Vec<&str> {
        self.args.iter().map(String::as_str).collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DeploymentRunSummary {
    container_name: String,
    container_id: Option<String>,
}

impl DeploymentRunSummary {
    fn completed_message(&self, sha: &str) -> String {
        match &self.container_id {
            Some(container_id) => format!(
                "✅ Automatic deployment of `{}` completed. Container `{}` is running as `{}`.",
                short_sha(sha),
                self.container_name,
                container_id
            ),
            None => format!(
                "✅ Automatic deployment of `{}` completed. Container `{}` is running.",
                short_sha(sha),
                self.container_name
            ),
        }
    }

    fn manual_completed_message(&self, sha: &str) -> String {
        match &self.container_id {
            Some(container_id) => format!(
                "✅ Deployment of `{}` completed. Container `{}` is running as `{}`.",
                short_sha(sha),
                self.container_name,
                container_id
            ),
            None => format!(
                "✅ Deployment of `{}` completed. Container `{}` is running.",
                short_sha(sha),
                self.container_name
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct GitHubCommit {
    pub sha: String,
    pub html_url: String,
    pub commit: GitHubCommitDetails,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct GitHubCommitDetails {
    pub message: String,
}

#[derive(Clone, Debug, Deserialize)]
struct GitHubComparison {
    #[serde(default)]
    commits: Vec<GitHubCommit>,
}

impl DeploymentBot {
    async fn cleanup_old_images(&self, sha: Option<&str>) {
        let main = sha
            .map(|sha| format!("ghcr.io/bushshrub/housebot:sha-{sha}"))
            .unwrap_or_else(|| "ghcr.io/bushshrub/housebot:latest".into());
        let previous = self.previous_image.read().await.clone();
        let mut keep = vec![main.as_str()];
        if let Some(previous) = previous.as_deref() {
            keep.push(previous);
        }
        if let Err(error) = cleanup_old_housebot_images(&keep).await {
            tracing::warn!(%error, "Could not clean up old housebot images");
        }
    }

    async fn rollback(&self) -> anyhow::Result<String> {
        let digest = self
            .previous_image
            .read()
            .await
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no previous image is available in this session"))?;
        let commands =
            container_commands_with_env(&digest, &self.docker_network, housebot_env(), false)?;

        for command in &commands {
            let output = run_docker(&command.args()).await?;
            if command.stage.is_health_check() && output != "true" {
                anyhow::bail!("house-chatbot is not running after rollback");
            }
        }
        Ok(format!("Rolled house-chatbot back to `{digest}`."))
    }

    async fn checkpoint_current_image(&self) -> anyhow::Result<()> {
        let image = run_docker(&[
            "inspect",
            "--format={{.Config.Image}}",
            HOUSE_CHATBOT_CONTAINER,
        ])
        .await;
        if let Ok(image) = image {
            if valid_housebot_image(&image) {
                *self.previous_image.write().await = Some(image);
            }
        }
        Ok(())
    }

    async fn commits(&self, sha: &str) -> anyhow::Result<(GitHubCommit, Vec<GitHubCommit>)> {
        let client = reqwest::Client::new();
        let base = format!("https://api.github.com/repos/{}", self.github_repo);
        let request = |url: String| {
            let request = client
                .get(url)
                .header("User-Agent", "housebot-deployment-bot")
                .header("Accept", "application/vnd.github+json");
            match &self.github_token {
                Some(token) => request.bearer_auth(token),
                None => request,
            }
        };
        let selected: GitHubCommit = request(format!("{base}/commits/{sha}"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let recent = request(format!("{base}/commits?sha={}&per_page=4", selected.sha))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok((selected, recent))
    }

    async fn latest_branch_commit(&self) -> anyhow::Result<GitHubCommit> {
        let client = reqwest::Client::new();
        let url = format!(
            "https://api.github.com/repos/{}/commits/{}",
            self.github_repo, self.github_branch
        );
        let request = client
            .get(url)
            .header("User-Agent", "housebot-deployment-bot")
            .header("Accept", "application/vnd.github+json");
        let request = match &self.github_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        };
        Ok(request.send().await?.error_for_status()?.json().await?)
    }

    async fn compare_commits(
        &self,
        current_sha: &str,
        target_sha: &str,
    ) -> anyhow::Result<Vec<GitHubCommit>> {
        let client = reqwest::Client::new();
        let url = format!(
            "https://api.github.com/repos/{}/compare/{}...{}",
            self.github_repo, current_sha, target_sha
        );
        let request = client
            .get(url)
            .header("User-Agent", "housebot-deployment-bot")
            .header("Accept", "application/vnd.github+json");
        let request = match &self.github_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        };
        Ok(request
            .send()
            .await?
            .error_for_status()?
            .json::<GitHubComparison>()
            .await?
            .commits)
    }

    async fn changelog(&self, current_sha: &str, target_sha: &str) -> anyhow::Result<String> {
        let commits = self.compare_commits(current_sha, target_sha).await?;
        Ok(deployment_changelog(current_sha, target_sha, &commits))
    }

    async fn current_running_sha(&self) -> anyhow::Result<String> {
        let image = run_docker(&["inspect", "--format={{.Config.Image}}", "house-chatbot"]).await?;
        let sha = image
            .strip_prefix("ghcr.io/bushshrub/housebot:sha-")
            .ok_or_else(|| {
                anyhow::anyhow!("running house-chatbot image does not contain a commit SHA")
            })?;
        if !valid_sha(sha) {
            anyhow::bail!("running house-chatbot image contains an invalid commit SHA");
        }
        Ok(sha.to_string())
    }

    async fn update_to_latest(&self) -> anyhow::Result<String> {
        let current_sha = match self.current_running_sha().await {
            Ok(sha) => Some(sha),
            Err(error) if docker_object_missing(&error) => None,
            Err(error) => return Err(error),
        };
        let latest = self.latest_branch_commit().await?;
        if current_sha.as_deref() == Some(latest.sha.as_str()) {
            return Ok(format!(
                "✅ Already running the latest `{}` commit on `{}`.",
                short_sha(current_sha.as_deref().expect("current SHA was checked")),
                self.github_branch
            ));
        }

        let _deployment_guard = self.deployment_lock.lock().await;
        let changelog = match current_sha.as_deref() {
            Some(current_sha) => self.changelog(current_sha, &latest.sha).await?,
            None => "**Changelog**\nNo previous housebot container was found; deploying the latest commit."
                .to_string(),
        };
        self.checkpoint_current_image().await?;
        let commands = deploy_commands(Some(&latest.sha), &self.docker_network)?;
        for command in &commands {
            tracing::info!(
                stage = %command.stage,
                "Update deployment stage started"
            );
            let output = run_docker(&command.args()).await?;
            if command.stage.is_health_check() && output != "true" {
                anyhow::bail!("house-chatbot is not running after update deployment");
            }
            tracing::info!(
                stage = %command.stage,
                "Update deployment stage completed"
            );
        }
        let previous = current_sha
            .as_deref()
            .map(short_sha)
            .unwrap_or("no running container");
        Ok(format!(
            "✅ Updated housebot from `{}` to latest `{}` on `{}`.\n\n{}",
            previous,
            short_sha(&latest.sha),
            self.github_branch,
            changelog
        ))
    }
}

pub fn valid_sha(sha: &str) -> bool {
    (7..=40).contains(&sha.len()) && sha.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn deploy_commands(
    sha: Option<&str>,
    docker_network: &str,
) -> anyhow::Result<Vec<DeploymentCommand>> {
    let main = match sha {
        Some(sha) if valid_sha(sha) => format!("ghcr.io/bushshrub/housebot:sha-{sha}"),
        Some(_) => anyhow::bail!("SHA must contain 7 to 40 hexadecimal characters"),
        None => "ghcr.io/bushshrub/housebot:latest".into(),
    };
    container_commands_with_env(&main, docker_network, housebot_env(), true)
}

fn valid_housebot_image(image: &str) -> bool {
    image.starts_with("ghcr.io/bushshrub/housebot:sha-")
        || image.starts_with("ghcr.io/bushshrub/housebot@sha256:")
}

pub fn container_commands(
    image: &str,
    docker_network: &str,
) -> anyhow::Result<Vec<DeploymentCommand>> {
    container_commands_with_env(image, docker_network, Vec::new(), false)
}

fn container_commands_with_env(
    image: &str,
    docker_network: &str,
    environment: Vec<(String, String)>,
    allow_latest: bool,
) -> anyhow::Result<Vec<DeploymentCommand>> {
    if !(valid_housebot_image(image)
        || (allow_latest && image == "ghcr.io/bushshrub/housebot:latest"))
    {
        anyhow::bail!("invalid housebot deployment image");
    }
    let mut run = vec![
        "run".into(),
        "--detach".into(),
        "--name".into(),
        HOUSE_CHATBOT_CONTAINER.into(),
        "--restart".into(),
        "unless-stopped".into(),
        "--network".into(),
        docker_network.into(),
    ];
    for (name, value) in environment {
        run.push("--env".into());
        run.push(format!("{name}={value}"));
    }
    run.extend(["--env".into(), "DATA_DIR=/app/data".into(), image.into()]);
    Ok(vec![
        DeploymentCommand::new(
            DeploymentStage::PullHousebotImage,
            vec!["pull".into(), image.into()],
        ),
        DeploymentCommand::new(
            DeploymentStage::RemovePreviousContainer,
            vec![
                "rm".into(),
                "--force".into(),
                HOUSE_CHATBOT_CONTAINER.into(),
            ],
        ),
        DeploymentCommand::new(DeploymentStage::StartRequestedImage, run),
        DeploymentCommand::new(
            DeploymentStage::CheckContainerState,
            vec![
                "inspect".into(),
                "--format={{.State.Running}}".into(),
                HOUSE_CHATBOT_CONTAINER.into(),
            ],
        ),
    ])
}

pub fn deploy_progress(stage: DeploymentStage) -> &'static str {
    stage.progress_message()
}

pub fn commit_summary(selected: &GitHubCommit, recent: &[GitHubCommit]) -> String {
    let first_line = selected
        .commit
        .message
        .lines()
        .next()
        .unwrap_or("No commit message");
    let mut text = format!(
        "Deploying [`{}`]({}) — {}\n\n**Recent commits:**",
        short_sha(&selected.sha),
        selected.html_url,
        first_line
    );
    for commit in recent
        .iter()
        .filter(|commit| commit.sha != selected.sha)
        .take(3)
    {
        let message = commit
            .commit
            .message
            .lines()
            .next()
            .unwrap_or("No commit message");
        text.push_str(&format!(
            "\n• [`{}`]({}) — {}",
            short_sha(&commit.sha),
            commit.html_url,
            message
        ));
    }
    text
}

pub fn deployment_changelog(
    current_sha: &str,
    target_sha: &str,
    commits: &[GitHubCommit],
) -> String {
    if commits.is_empty() {
        return format!(
            "**Changelog**\n`{}` → `{}`\nNo commits found between these deployments.",
            short_sha(current_sha),
            short_sha(target_sha)
        );
    }

    let mut text = format!(
        "**Changelog since `{}`** ({} commit{})",
        short_sha(current_sha),
        commits.len(),
        if commits.len() == 1 { "" } else { "s" }
    );
    for (shown, commit) in commits.iter().enumerate() {
        let message = commit
            .commit
            .message
            .lines()
            .next()
            .unwrap_or("No commit message");
        let line = format!(
            "\n• [`{}`]({}) — {}",
            short_sha(&commit.sha),
            commit.html_url,
            message
        );
        if text.len() + line.len() > 1_800 {
            text.push_str(&format!(
                "\n• …and {} more commit(s)",
                commits.len() - shown
            ));
            break;
        }
        text.push_str(&line);
    }
    text
}

fn short_sha(sha: &str) -> &str {
    sha.get(..7).unwrap_or(sha)
}

fn docker_object_missing(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.contains("No such object") || message.contains("No such container")
}

async fn run_docker(args: &[&str]) -> anyhow::Result<String> {
    let output = ProcessCommand::new("docker")
        .args(args)
        .current_dir("/")
        .output()
        .await?;
    let missing_container = args.first() == Some(&"rm")
        && (String::from_utf8_lossy(&output.stderr).contains("No such container")
            || String::from_utf8_lossy(&output.stderr).contains("No such object"));
    if !output.status.success() && !missing_container {
        anyhow::bail!(
            "docker command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

async fn cleanup_old_housebot_images(keep: &[&str]) -> anyhow::Result<()> {
    let images = run_docker(&[
        "images",
        "--format={{.Repository}}:{{.Tag}}",
        "ghcr.io/bushshrub/housebot*",
    ])
    .await?;
    for image in images.lines().filter(|image| {
        (*image == "ghcr.io/bushshrub/housebot:latest"
            || image.starts_with("ghcr.io/bushshrub/housebot:sha-"))
            && !keep.contains(image)
    }) {
        run_docker(&["image", "rm", image]).await?;
    }
    Ok(())
}

const HOUSEBOT_ENV_VARS: &[&str] = &[
    "DISCORD_BOT_TOKEN",
    "OWNER_DISCORD_ID",
    "DATABASE_URL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "MAX_HISTORY_TURNS",
    "MAX_CONTEXT_TOKENS",
    "CONVERSATION_IDLE_TIMEOUT",
    "JELLYFIN_URL",
    "JELLYFIN_API_KEY",
    "LLAMA_CPP_URL",
    "LLAMA_CPP_MODEL",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_INSTALLATION_ID",
    "GITHUB_REPO",
];

fn housebot_env() -> Vec<(String, String)> {
    let mut values: HashMap<String, String> = HOUSEBOT_ENV_VARS
        .iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| ((*name).into(), value))
        })
        .collect();

    // Read the mounted deployment configuration at deploy time. This lets an
    // edited .env take effect without restarting the deployment bot itself.
    for path in ["/app/.env", ".env"] {
        if let Ok(contents) = std::fs::read_to_string(path) {
            for (name, value) in parse_dotenv(&contents) {
                if HOUSEBOT_ENV_VARS.contains(&name.as_str()) {
                    values.insert(name, value);
                }
            }
        }
    }

    HOUSEBOT_ENV_VARS
        .iter()
        .filter_map(|name| values.remove(*name).map(|value| ((*name).into(), value)))
        .collect()
}

fn parse_dotenv(contents: &str) -> Vec<(String, String)> {
    contents
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix("export ").unwrap_or(line);
            let (name, value) = line.split_once('=')?;
            let name = name.trim();
            if name.is_empty() || name.starts_with('#') {
                return None;
            }
            let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
            Some((name.to_string(), value.to_string()))
        })
        .collect()
}

#[serenity::async_trait]
impl EventHandler for DeploymentBot {
    async fn ready(&self, ctx: Context, ready: Ready) {
        tracing::info!("Deployment bot logged in as {}", ready.user.name);
        let commands = deployment_commands();
        if let Some(guild_id) = self.guild_id {
            remove_global_deployment_commands(&ctx).await;
            if let Err(error) = GuildId::new(guild_id)
                .set_commands(&ctx.http, commands)
                .await
            {
                tracing::error!(
                    guild_id,
                    "Failed to sync deployment slash commands: {error}"
                );
            } else {
                tracing::info!(guild_id, "Synced deployment slash commands to guild");
            }
        } else {
            for command in commands {
                if let Err(error) = Command::create_global_command(&ctx.http, command).await {
                    tracing::error!("Failed to register deployment slash command: {error}");
                }
            }
        }
    }

    async fn message(&self, ctx: Context, message: Message) {
        if message.channel_id.get() != self.channel_id {
            return;
        }
        if let Some(event) = deployment_event(&message) {
            tracing::info!(succeeded = event.succeeded, commit = ?event.commit, "Observed deployment webhook");
            let Some(sha) = event.commit.clone().filter(|_| event.succeeded) else {
                return;
            };
            if !valid_sha(&sha) {
                tracing::error!("Deployment webhook contained an invalid SHA");
                return;
            }
            let _deployment_guard = self.deployment_lock.lock().await;
            if self
                .last_event
                .read()
                .await
                .as_ref()
                .is_some_and(|previous| {
                    previous.succeeded && previous.commit.as_deref() == Some(&sha)
                })
            {
                tracing::info!(sha, "Ignoring duplicate build notification");
                return;
            }
            if let Err(error) = self.checkpoint_current_image().await {
                tracing::error!("Could not save deployment checkpoint: {error}");
                return;
            }
            let changelog = match self.current_running_sha().await {
                Ok(current_sha) => match self.changelog(&current_sha, &sha).await {
                    Ok(changelog) => Some(changelog),
                    Err(error) => {
                        tracing::warn!(%error, "Could not build deployment changelog");
                        None
                    }
                },
                Err(error) => {
                    tracing::warn!(%error, "Could not determine previous deployed commit");
                    None
                }
            };
            let commands = match deploy_commands(Some(&sha), &self.docker_network) {
                Ok(commands) => commands,
                Err(error) => {
                    tracing::error!("Could not prepare deployment: {error}");
                    return;
                }
            };
            let mut summary = DeploymentRunSummary {
                container_name: HOUSE_CHATBOT_CONTAINER.into(),
                container_id: None,
            };
            for command in &commands {
                tracing::info!(
                    stage = %command.stage,
                    "Automatic deployment progress"
                );
                let _ = message
                    .channel_id
                    .say(&ctx.http, command.stage.progress_message())
                    .await;
                match run_docker(&command.args()).await {
                    Ok(output) if command.stage.is_health_check() && output != "true" => {
                        tracing::error!(
                            stage = %command.stage,
                            "Automatic deployment stage failed: house-chatbot is not running"
                        );
                        return;
                    }
                    Ok(output) => {
                        if command.stage.is_start() {
                            summary.container_id = Some(output);
                        }
                        tracing::info!(
                            stage = %command.stage,
                            "Automatic deployment stage completed"
                        );
                    }
                    Err(error) => {
                        tracing::error!(
                            stage = %command.stage,
                            "Automatic deployment stage failed: {error}"
                        );
                        return;
                    }
                }
            }
            self.cleanup_old_images(Some(&sha)).await;
            tracing::info!(sha, container = %summary.container_name, container_id = ?summary.container_id, "Automatic deployment completed");
            *self.last_event.write().await = Some(event);
            if let Some(changelog) = changelog {
                let _ = message.channel_id.say(&ctx.http, changelog).await;
            }
            let _ = message
                .channel_id
                .say(&ctx.http, summary.completed_message(&sha))
                .await;
        }
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Component(component) = interaction {
            if component.data.custom_id == "deploy_deny" {
                let response = CreateInteractionResponse::UpdateMessage(
                    CreateInteractionResponseMessage::new()
                        .content("Deployment cancelled.")
                        .components(vec![]),
                );
                let _ = component.create_response(&ctx.http, response).await;
                return;
            }
            let Some(sha) = component.data.custom_id.strip_prefix("deploy_confirm:") else {
                return;
            };
            if !rollback_allowed(
                self.owner_id,
                component.user.id.get(),
                component.channel_id.get(),
                self.channel_id,
            ) {
                let response = CreateInteractionResponse::Message(
                    CreateInteractionResponseMessage::new()
                        .content("You are not allowed to deploy.")
                        .ephemeral(true),
                );
                let _ = component.create_response(&ctx.http, response).await;
                return;
            }
            let response = CreateInteractionResponse::UpdateMessage(
                CreateInteractionResponseMessage::new()
                    .content("⬇️ Starting deployment…")
                    .components(vec![]),
            );
            if component
                .create_response(&ctx.http, response)
                .await
                .is_err()
            {
                return;
            }
            let commands = if sha == "latest" {
                deploy_commands(None, &self.docker_network)
            } else {
                deploy_commands(Some(sha), &self.docker_network)
            };
            let result = async {
                let _deployment_guard = self.deployment_lock.lock().await;
                self.checkpoint_current_image().await?;
                let commands = commands?;
                let mut summary = DeploymentRunSummary {
                    container_name: HOUSE_CHATBOT_CONTAINER.into(),
                    container_id: None,
                };
                for command in &commands {
                    tracing::info!(
                        stage = %command.stage,
                        "Manual deployment stage started"
                    );
                    component
                        .edit_response(
                            &ctx.http,
                            EditInteractionResponse::new()
                                .content(command.stage.progress_message()),
                        )
                        .await?;
                    let output = match run_docker(&command.args()).await {
                        Ok(output) => output,
                        Err(error) => {
                            tracing::error!(
                                stage = %command.stage,
                                "Manual deployment stage failed: {error}"
                            );
                            return Err(error);
                        }
                    };
                    if command.stage.is_health_check() && output != "true" {
                        anyhow::bail!(
                            "deployment stage `{}` failed: house-chatbot is not running",
                            command.stage
                        );
                    }
                    if command.stage.is_start() {
                        summary.container_id = Some(output);
                    }
                    tracing::info!(
                        stage = %command.stage,
                        "Manual deployment stage completed"
                    );
                }
                self.cleanup_old_images((sha != "latest").then_some(sha))
                    .await;
                anyhow::Ok(summary)
            }
            .await;
            if let Err(error) = &result {
                tracing::error!("Manual deployment failed: {error}");
            }
            let content = match result {
                Ok(summary) => summary.manual_completed_message(sha),
                Err(error) => format!("❌ Deployment failed: {error}"),
            };
            let _ = component
                .edit_response(&ctx.http, EditInteractionResponse::new().content(content))
                .await;
            return;
        }
        let Interaction::Command(command) = interaction else {
            return;
        };
        if command.data.name == "deploy" {
            if !rollback_allowed(
                self.owner_id,
                command.user.id.get(),
                command.channel_id.get(),
                self.channel_id,
            ) {
                let response = CreateInteractionResponse::Message(
                    CreateInteractionResponseMessage::new()
                        .content("Only the configured owner can deploy from this channel.")
                        .ephemeral(true),
                );
                let _ = command.create_response(&ctx.http, response).await;
                return;
            }
            let sha = match command.data.options.first().map(|option| &option.value) {
                Some(CommandDataOptionValue::String(sha)) if valid_sha(sha) => Some(sha.as_str()),
                Some(_) => {
                    let response = CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("SHA must contain 7 to 40 hexadecimal characters.")
                            .ephemeral(true),
                    );
                    let _ = command.create_response(&ctx.http, response).await;
                    return;
                }
                None => None,
            };
            let response = match match sha {
                Some(sha) => self.commits(sha).await,
                None => self
                    .latest_branch_commit()
                    .await
                    .map(|latest| (latest, Vec::new())),
            } {
                Ok((selected, recent)) => {
                    let description = match self.current_running_sha().await {
                        Ok(current_sha) => {
                            match self.changelog(&current_sha, &selected.sha).await {
                                Ok(changelog) => format!(
                                    "{}\n\n{}",
                                    commit_summary(&selected, &recent),
                                    changelog
                                ),
                                Err(error) => format!(
                                    "{}\n\nChangelog unavailable: {error}",
                                    commit_summary(&selected, &recent)
                                ),
                            }
                        }
                        Err(error) => format!(
                            "{}\n\nChangelog unavailable: {error}",
                            commit_summary(&selected, &recent)
                        ),
                    };
                    CreateInteractionResponseMessage::new()
                        .embed(
                            CreateEmbed::new()
                                .title("Confirm deployment")
                                .description(description),
                        )
                        .components(vec![CreateActionRow::Buttons(vec![
                            CreateButton::new(format!(
                                "deploy_confirm:{}",
                                sha.unwrap_or("latest")
                            ))
                            .label("Confirm")
                            .style(ButtonStyle::Success),
                            CreateButton::new("deploy_deny")
                                .label("Deny")
                                .style(ButtonStyle::Danger),
                        ])])
                        .ephemeral(true)
                }
                Err(error) => CreateInteractionResponseMessage::new()
                    .content(format!("Could not find that commit: {error}"))
                    .ephemeral(true),
            };
            let _ = command
                .create_response(&ctx.http, CreateInteractionResponse::Message(response))
                .await;
            return;
        }
        if command.data.name == "update" {
            if !rollback_allowed(
                self.owner_id,
                command.user.id.get(),
                command.channel_id.get(),
                self.channel_id,
            ) {
                let response = CreateInteractionResponse::Message(
                    CreateInteractionResponseMessage::new()
                        .content("Only the configured owner can update from this channel.")
                        .ephemeral(true),
                );
                let _ = command.create_response(&ctx.http, response).await;
                return;
            }
            let response = CreateInteractionResponse::Message(
                CreateInteractionResponseMessage::new()
                    .content("🔎 Checking the running commit against the branch tip…")
                    .ephemeral(true),
            );
            if command.create_response(&ctx.http, response).await.is_err() {
                return;
            }
            let content = match self.update_to_latest().await {
                Ok(message) => message,
                Err(error) => format!("❌ Update failed: {error}"),
            };
            let _ = command
                .edit_response(&ctx.http, EditInteractionResponse::new().content(content))
                .await;
            return;
        }
        if command.data.name != "rollback" {
            return;
        }
        let allowed = rollback_allowed(
            self.owner_id,
            command.user.id.get(),
            command.channel_id.get(),
            self.channel_id,
        );
        let reply = if !allowed {
            "Only the configured owner can roll back from the deployment channel.".to_string()
        } else {
            let _deployment_guard = self.deployment_lock.lock().await;
            match self.rollback().await {
                Ok(message) => message,
                Err(error) => format!("Rollback failed: {error}"),
            }
        };
        let response = CreateInteractionResponse::Message(
            CreateInteractionResponseMessage::new()
                .content(reply)
                .ephemeral(true),
        );
        if let Err(error) = command.create_response(&ctx.http, response).await {
            tracing::warn!("Failed to respond to /rollback: {error}");
        }
    }
}

pub async fn run() -> anyhow::Result<()> {
    let token = std::env::var("DEPLOYMENT_DISCORD_BOT_TOKEN")
        .map_err(|_| anyhow::anyhow!("DEPLOYMENT_DISCORD_BOT_TOKEN is not set"))?;
    let owner_id = env_u64("OWNER_DISCORD_ID")?;
    let channel_id = env_u64("DEPLOYMENT_CHANNEL_ID")?;
    let guild_id = optional_env_u64("DEPLOYMENT_GUILD_ID")?;
    let handler = DeploymentBot {
        owner_id,
        channel_id,
        guild_id,
        last_event: Arc::new(RwLock::new(None)),
        previous_image: Arc::new(RwLock::new(None)),
        deployment_lock: Arc::new(Mutex::new(())),
        github_repo: std::env::var("GITHUB_REPO").unwrap_or_else(|_| "bushshrub/housebot".into()),
        github_branch: std::env::var("GITHUB_BRANCH").unwrap_or_else(|_| "master".into()),
        github_token: std::env::var("GITHUB_TOKEN").ok(),
        docker_network: std::env::var("DOCKER_NETWORK")
            .unwrap_or_else(|_| "house-chatbot_default".into()),
    };
    let intents = GatewayIntents::non_privileged() | GatewayIntents::MESSAGE_CONTENT;
    let mut client = Client::builder(token, intents)
        .event_handler(handler)
        .await?;

    tokio::select! {
        result = client.start() => result?,
        _ = shutdown_signal() => {
            tracing::info!("Deployment bot shutting down and disconnecting from Discord");
            shutdown_main_bot().await;
        }
    }
    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut terminate = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = terminate.recv() => {},
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn shutdown_main_bot() {
    tracing::info!("Stopping main house-chatbot container");
    if let Err(error) = run_docker(&["stop", "--time", "10", HOUSE_CHATBOT_CONTAINER]).await {
        tracing::warn!("Could not stop main house-chatbot container: {error}");
    }
    if let Err(error) = run_docker(&["rm", "--force", HOUSE_CHATBOT_CONTAINER]).await {
        tracing::warn!("Could not remove main house-chatbot container: {error}");
    }
    tracing::info!("Main house-chatbot container stopped");
}

fn env_u64(name: &str) -> anyhow::Result<u64> {
    std::env::var(name)
        .map_err(|_| anyhow::anyhow!("{name} is not set"))?
        .parse()
        .map_err(|_| anyhow::anyhow!("{name} must be a Discord numeric ID"))
}

fn optional_env_u64(name: &str) -> anyhow::Result<Option<u64>> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .parse()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("{name} must be a Discord numeric ID")),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn deployment_commands() -> Vec<CreateCommand> {
    vec![
        CreateCommand::new("rollback")
            .description("Roll back housebot to the previous deployed image"),
        CreateCommand::new("update")
            .description("Redeploy the latest commit from the configured branch"),
        CreateCommand::new("deploy")
            .description("Deploy a previously built commit")
            .add_option(
                CreateCommandOption::new(CommandOptionType::String, "sha", "Git commit SHA")
                    .required(false),
            ),
    ]
}

async fn remove_global_deployment_commands(ctx: &Context) {
    let commands = match Command::get_global_commands(&ctx.http).await {
        Ok(commands) => commands,
        Err(error) => {
            tracing::error!("Failed to inspect global deployment slash commands: {error}");
            return;
        }
    };

    for command in commands.into_iter().filter(|command| {
        command.name == "deploy" || command.name == "rollback" || command.name == "update"
    }) {
        if let Err(error) = Command::delete_global_command(&ctx.http, command.id).await {
            tracing::error!(name = %command.name, "Failed to remove global deployment slash command: {error}");
        } else {
            tracing::info!(name = %command.name, "Removed global deployment slash command");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollback_is_owner_and_channel_scoped() {
        assert!(rollback_allowed(10, 10, 20, 20));
        assert!(!rollback_allowed(10, 11, 20, 20));
        assert!(!rollback_allowed(10, 10, 21, 20));
        assert!(!rollback_allowed(0, 0, 20, 20));
    }

    #[test]
    fn invalid_numeric_environment_value_is_rejected() {
        std::env::set_var("DEPLOYMENT_BOT_TEST_ID", "not-a-number");
        let error = env_u64("DEPLOYMENT_BOT_TEST_ID").unwrap_err().to_string();
        std::env::remove_var("DEPLOYMENT_BOT_TEST_ID");
        assert!(error.contains("numeric ID"));
    }

    #[test]
    fn optional_guild_id_accepts_unset_and_numeric_values() {
        std::env::remove_var("DEPLOYMENT_BOT_TEST_GUILD_ID");
        assert_eq!(
            optional_env_u64("DEPLOYMENT_BOT_TEST_GUILD_ID").unwrap(),
            None
        );

        std::env::set_var("DEPLOYMENT_BOT_TEST_GUILD_ID", "123456789");
        assert_eq!(
            optional_env_u64("DEPLOYMENT_BOT_TEST_GUILD_ID").unwrap(),
            Some(123456789)
        );
        std::env::remove_var("DEPLOYMENT_BOT_TEST_GUILD_ID");
    }

    #[test]
    fn deployment_webhook_text_is_classified_strictly() {
        assert_eq!(
            classify_deployment_text("HomeLab deployment succeeded"),
            Some(true)
        );
        assert_eq!(
            classify_deployment_text("HomeLab deployment FAILED"),
            Some(false)
        );
        assert_eq!(classify_deployment_text("build succeeded"), Some(true));
        assert_eq!(classify_deployment_text("tests succeeded"), None);
    }

    #[test]
    fn rollback_plan_uses_only_the_checkpoint_digest() {
        let digest = "ghcr.io/bushshrub/housebot@sha256:abc123";
        let commands = container_commands(digest, "network").unwrap();
        assert_eq!(commands.len(), 4);
        assert_eq!(commands[0].stage, DeploymentStage::PullHousebotImage);
        assert_eq!(commands[0].args, vec!["pull", digest]);
        assert_eq!(commands[2].stage, DeploymentStage::StartRequestedImage);
        assert_eq!(commands[2].args.last().unwrap(), digest);
    }

    #[test]
    fn rollback_rejects_tags_and_unrelated_images() {
        assert!(container_commands("ghcr.io/bushshrub/housebot:latest", "network").is_err());
        assert!(container_commands("ghcr.io/other/image@sha256:abc", "network").is_err());
        assert!(container_commands("none", "network").is_err());
    }

    #[test]
    fn deploy_plan_is_sha_scoped_and_rejects_injection() {
        let commands = deploy_commands(Some("abcdef123456"), "network").unwrap();
        assert_eq!(commands.len(), 4);
        assert_eq!(
            commands
                .iter()
                .map(|command| command.stage)
                .collect::<Vec<_>>(),
            vec![
                DeploymentStage::PullHousebotImage,
                DeploymentStage::RemovePreviousContainer,
                DeploymentStage::StartRequestedImage,
                DeploymentStage::CheckContainerState,
            ]
        );
        assert!(commands[0].args[1].ends_with(":sha-abcdef123456"));
        assert!(!commands[3].args.contains(&"/deployment".to_string()));
        assert_eq!(
            deploy_commands(None, "network").unwrap()[0].args[1],
            "ghcr.io/bushshrub/housebot:latest"
        );
        assert!(deploy_commands(Some("latest"), "network").is_err());
        assert!(deploy_commands(Some("abcdef;reboot"), "network").is_err());
    }

    #[test]
    fn completed_deployment_message_includes_container_name_and_id() {
        let summary = DeploymentRunSummary {
            container_name: HOUSE_CHATBOT_CONTAINER.into(),
            container_id: Some("abc123def456".into()),
        };

        let message = summary.completed_message("abcdef123456");

        assert!(message.contains("Container `house-chatbot`"));
        assert!(message.contains("`abc123def456`"));
    }

    #[test]
    fn commit_summary_has_links_messages_and_alternatives() {
        let commit = |sha: &str, message: &str| GitHubCommit {
            sha: sha.into(),
            html_url: format!("https://github.com/example/repo/commit/{sha}"),
            commit: GitHubCommitDetails {
                message: message.into(),
            },
        };
        let selected = commit("abcdef1234", "selected commit\nbody");
        let summary = commit_summary(
            &selected,
            &[selected.clone(), commit("1234567890", "older")],
        );
        assert!(summary.contains("[`abcdef1`](https://github.com/example/repo/commit/abcdef1234)"));
        assert!(summary.contains("selected commit"));
        assert!(summary.contains("older"));
    }

    #[test]
    fn deployment_changelog_lists_commits_since_previous_deployment() {
        let commit = |sha: &str, message: &str| GitHubCommit {
            sha: sha.into(),
            html_url: format!("https://github.com/example/repo/commit/{sha}"),
            commit: GitHubCommitDetails {
                message: message.into(),
            },
        };
        let changelog = deployment_changelog(
            "1111111",
            "3333333",
            &[commit("2222222", "Add deployment visibility\nDetails")],
        );
        assert!(changelog.contains("since `1111111`"));
        assert!(changelog.contains("1 commit"));
        assert!(changelog.contains("Add deployment visibility"));
        assert!(changelog.contains("https://github.com/example/repo/commit/2222222"));
    }
}
