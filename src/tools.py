from ddgs import AsyncDDGS
import asyncio
from typing import List, TypedDict, Annotated

class RealTimeAgent:
    def __init__(self):
        self.timeout = 10 

    async def _search(self, query: str, max_results: int = 5) -> str:
        try:
            async with AsyncDDGS(timeout=self.timeout) as ddgs:
                results = []
                async for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
                
                if not results:
                    return f"No recent info found for: {query}"
                
                return "\n".join([f"- {r['title']}: {r['href']}\n  {r['body']}" for r in results])
        
        except Exception as e:
            return f"Search timed out or was blocked for query: {query}"

   
    async def get_history(self, target: str):
        return await self._search(f"history and historical importance of {target}")

    async def get_logistics(self, target: str):
        return await self._search(f"how to travel to {target} public transport parking")

    async def get_culture(self, target: str):
        return await self._search(f"local etiquette, safety tips, and current trends {target}")
        
    async def get_accommodation(self, target: str):
        return await self._search(f"best hotels and hostels near {target}")
        
    async def get_food(self, target: str):
        return await self._search(f"best local food and restaurants near {target}")

# Export the initialized agent
rt_tools = RealTimeAgent()