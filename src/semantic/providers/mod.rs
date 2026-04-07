//! LLM provider implementations

pub mod openai;
pub mod anthropic;
pub mod groq;
pub mod openrouter;

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// Trait for LLM providers that generate structured query responses
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a prompt and get response
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt to send to the LLM
    /// * `json_mode` - Whether to request JSON structured output (true) or plain text (false)
    ///
    /// When `json_mode` is true, the response should be valid JSON matching the QueryResponse schema.
    /// When `json_mode` is false, the response can be plain text (used for answer generation).
    async fn complete(&self, prompt: &str, json_mode: bool) -> Result<String>;

    /// Get provider name (for logging and error messages)
    fn name(&self) -> &str;

    /// Get default model identifier
    fn default_model(&self) -> &str;
}

/// Create a provider instance from name and API key
///
/// The `options` parameter allows passing provider-specific settings.
/// Currently used by OpenRouter for sort strategy (e.g., `{"sort": "price"}`).
/// Other providers ignore this parameter.
pub fn create_provider(
    provider_name: &str,
    api_key: String,
    model: Option<String>,
    options: Option<HashMap<String, String>>,
) -> Result<Box<dyn LlmProvider>> {
    match provider_name.to_lowercase().as_str() {
        "openai" => Ok(Box::new(openai::OpenAiProvider::new(api_key, model)?)),
        "anthropic" => Ok(Box::new(anthropic::AnthropicProvider::new(api_key, model)?)),
        "groq" => Ok(Box::new(groq::GroqProvider::new(api_key, model)?)),
        "openrouter" => {
            let sort = options.as_ref().and_then(|o| o.get("sort").cloned());
            Ok(Box::new(openrouter::OpenRouterProvider::new(api_key, model, sort)?))
        }
        _ => anyhow::bail!(
            "Unknown provider: {}. Supported: openai, anthropic, groq, openrouter",
            provider_name
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_provider_openai() {
        let provider = create_provider("openai", "test-key".to_string(), None, None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "openai");
    }

    #[test]
    fn test_create_provider_case_insensitive() {
        let provider = create_provider("OpenAI", "test-key".to_string(), None, None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_create_provider_unknown() {
        let provider = create_provider("unknown", "test-key".to_string(), None, None);
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("Unknown provider"));
        }
    }

    #[test]
    fn test_create_provider_openrouter() {
        let provider = create_provider("openrouter", "test-key".to_string(), None, None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "openrouter");
    }

    #[test]
    fn test_create_provider_openrouter_with_sort() {
        let mut opts = HashMap::new();
        opts.insert("sort".to_string(), "speed".to_string());
        let provider = create_provider(
            "openrouter",
            "test-key".to_string(),
            Some("openai/gpt-4o-mini".to_string()),
            Some(opts),
        );
        assert!(provider.is_ok());
    }
}
