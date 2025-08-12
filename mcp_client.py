from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

load_dotenv(override=True)


server = MCPServerSSE(url="http://127.0.0.1:8000/sse")
agent = Agent("google-gla:gemini-2.5-flash", toolsets=[server])


async def main():
    async with agent:
        result = await agent.run("Tell me  about BERT-based models .. ")
    print(result.output)


# run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
