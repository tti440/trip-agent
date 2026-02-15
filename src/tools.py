from ddgs import DDGS

class RealTimeAgent:
    def __init__(self):
        self.ddgs = DDGS()

    def _search(self, query, max_results=10):
        """Helper to run the search and format results."""
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            return "\n".join([f"- {r['title']}: {r['href']}\n  {r['body']}" for r in results])
        except Exception as e:
            return f"Search failed: {e}"

    def get_history(self, target):
        return self._search(f"history and significance of {target}")

    def get_logistics(self, target):
        return self._search(f"public transport and how to get to {target} parking")

    def get_culture(self, target):
        return self._search(f"tourist etiquette, safety tips, and current trends {target}")
        
    def get_accommodation(self, target):
        return self._search(f"best hotels and hostels near {target}")
        
    def get_food(self, target):
        return self._search(f"best restaurants and street food near {target}")

# Initialize and export
rt_tools = RealTimeAgent()