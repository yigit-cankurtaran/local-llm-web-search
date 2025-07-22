import argparse
from abc import ABC, abstractmethod
from duckduckgo_search import DDGS

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def chat(self, messages: list, system_prompt: str = None) -> str:
        """Send messages to the LLM and return response"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""
    
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model
        try:
            from ollama import chat
            self.chat_fn = chat
        except ImportError:
            raise ImportError("ollama package required for Ollama provider. Install with: pip install ollama")
    
    def chat(self, messages: list, system_prompt: str = None) -> str:
        if system_prompt and messages:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = self.chat_fn(model=self.model, messages=messages)
        return response["message"]["content"]

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required for OpenAI provider. Install with: pip install openai")
    
    def chat(self, messages: list, system_prompt: str = None) -> str:
        if system_prompt and messages:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None):
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required for Anthropic provider. Install with: pip install anthropic")
    
    def chat(self, messages: list, system_prompt: str = None) -> str:
        # Convert messages format for Anthropic
        formatted_messages = []
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append(msg)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt if system_prompt else "You are a helpful AI assistant.",
            messages=formatted_messages
        )
        return response.content[0].text

# basic wrapper to search ddg
def search_ddg(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return [r['body'] for r in results][:max_results]

# ask LLM if we need to search
def should_search(prompt, llm_provider):
    response = llm_provider.chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="you are an AI assistant that decides if a prompt needs an internet search. answer only YES or NO."
    )
    return "yes" in response.lower()

# summarize search results with LLM
def summarize_with_llm(query, results, llm_provider):
    text = "\n".join(f"- {r}" for r in results)
    response = llm_provider.chat(
        messages=[{"role": "user", "content": f"here are the search results about '{query}':\n{text}\n\nplease summarize them."}],
        system_prompt="you are an AI assistant that summarizes search results."
    )
    return response

# full agent
def ask_agent(prompt, llm_provider):
    if should_search(prompt, llm_provider):
        results = search_ddg(prompt)
        print("\n🔍 top search results:\n")
        for i, r in enumerate(results):
            print(f"{i+1}. {r}\n")

        # ask LLM to summarize AND explain its thought process
        full_input = "\n".join(f"- {r}" for r in results)
        response = llm_provider.chat(
            messages=[{"role": "user", "content": f"""you searched for: {prompt}
here are the results:
{full_input}

first, think out loud about what these say.
then, give a numbered summary starting with: 'to summarize the search results:'"""}],
            system_prompt="you are an AI assistant that receives search results and thinks aloud before summarizing them."
        )
        return response
    else:
        response = llm_provider.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        return response

def create_llm_provider(provider_name, model, api_key=None):
    """Factory function to create LLM providers"""
    if provider_name == "ollama":
        return OllamaProvider(model)
    elif provider_name == "openai":
        return OpenAIProvider(model, api_key)
    elif provider_name == "anthropic":
        return AnthropicProvider(model, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: ollama, openai, anthropic")

def main():
    parser = argparse.ArgumentParser(description="Local LLM Web Search Agent")
    parser.add_argument("--provider", choices=["ollama", "openai", "anthropic"], 
                       default="ollama", help="LLM provider to use")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--api-key", help="API key for cloud providers")
    
    args = parser.parse_args()
    
    # Set default models for each provider
    default_models = {
        "ollama": "qwen3:0.6b",
        "openai": "gpt-3.5-turbo", 
        "anthropic": "claude-3-haiku-20240307"
    }
    
    model = args.model or default_models[args.provider]
    
    try:
        llm_provider = create_llm_provider(args.provider, model, args.api_key)
        print(f"🤖 Using {args.provider} with model: {model}")
        
        while True:
            prompt = input("\nask > ")
            if prompt.lower() in ['quit', 'exit']:
                break
            print(ask_agent(prompt, llm_provider))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
