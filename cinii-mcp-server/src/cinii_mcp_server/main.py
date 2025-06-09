"""Entry point for the CiNii MCP server."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from mcp.server import MCPServer, Tool
from pydantic import BaseModel, Field

# Retrieve the CiNii API key from the environment
CII_APPID = os.getenv("CII_APPID")

XML_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "openSearch": "http://a9.com/-/spec/opensearch/1.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "prism": "http://prismstandard.org/namespaces/basic/2.1/",
    "cinii": "http://ci.nii.ac.jp/ns/1.0/",
}


class CiNiiArticleSearchInput(BaseModel):
    """Request model for searching articles."""

    query: str = Field(..., description="Search term")
    count: int = Field(10, description="Number of results", le=200)
    start: int = Field(1, description="Offset for pagination", ge=1)


class CiNiiArticle(BaseModel):
    title: Optional[str] = None
    link: Optional[str] = None
    author: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    page_start: Optional[str] = None
    page_end: Optional[str] = None
    publication_year: Optional[str] = None
    description: Optional[str] = None
    crid: Optional[str] = None


class CiNiiArticleSearchResult(BaseModel):
    articles: List[CiNiiArticle]
    total_results: int
    start_index: int
    items_per_page: int


class CiNiiClient:
    """Client for the CiNii OpenSearch API."""

    BASE_URL = "https://ci.nii.ac.jp/opensearch/articles"

    def __init__(self, appid: str) -> None:
        if not appid:
            raise ValueError(
                "CII_APPID environment variable is required for CiNii API access"
            )
        self.appid = appid
        self.client = httpx.AsyncClient()

    async def close(self) -> None:
        await self.client.aclose()

    def _text(self, element: ET.Element, path: str) -> Optional[str]:
        found = element.find(path, XML_NS)
        return found.text if found is not None else None

    def _int(self, element: ET.Element, path: str) -> int:
        value = self._text(element, path)
        return int(value) if value and value.isdigit() else 0

    def _parse_article(self, entry: ET.Element) -> CiNiiArticle:
        authors = [a.text for a in entry.findall("atom:author/atom:name", XML_NS) if a.text]
        link_el = entry.find("atom:link", XML_NS)
        link = link_el.attrib.get("href") if link_el is not None else None
        pub_date = self._text(entry, "prism:publicationDate")
        publication_year = pub_date.split("-")[0] if pub_date else None
        return CiNiiArticle(
            title=self._text(entry, "atom:title"),
            author=", ".join(authors) if authors else None,
            journal=self._text(entry, "prism:publicationName"),
            volume=self._text(entry, "prism:volume"),
            issue=self._text(entry, "prism:number"),
            page_start=self._text(entry, "prism:startingPage"),
            page_end=self._text(entry, "prism:endingPage"),
            publication_year=publication_year,
            link=link,
            description=self._text(entry, "atom:summary"),
            crid=self._text(entry, "cinii:crid"),
        )

    async def search_articles(
        self, query: str, count: int = 10, start: int = 1
    ) -> CiNiiArticleSearchResult:
        params = {
            "q": query,
            "count": count,
            "start": start,
            "format": "atom",
            "appid": self.appid,
        }
        try:
            r = await self.client.get(self.BASE_URL, params=params)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            articles = [self._parse_article(e) for e in root.findall("atom:entry", XML_NS)]
            return CiNiiArticleSearchResult(
                articles=articles,
                total_results=self._int(root, "openSearch:totalResults"),
                start_index=self._int(root, "openSearch:startIndex"),
                items_per_page=self._int(root, "openSearch:itemsPerPage"),
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except ET.ParseError:
            raise HTTPException(status_code=500, detail="Failed to parse CiNii XML response")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


cinii_client: Optional[CiNiiClient] = None
if CII_APPID:
    cinii_client = CiNiiClient(appid=CII_APPID)
else:
    print(
        "WARNING: CII_APPID environment variable not set. CiNii tools will be disabled."
    )

cinii_tools = []
if cinii_client:
    cinii_tools.append(
        Tool(
            name="search_articles",
            description="Search academic articles on CiNii",
            input_model=CiNiiArticleSearchInput,
            output_model=CiNiiArticleSearchResult,
            function=cinii_client.search_articles,
        )
    )

mcp_server = MCPServer(
    name="cinii",
    description="MCP server providing access to CiNii articles",
    tools=cinii_tools,
)

app = FastAPI(title="CiNii MCP Server", version="0.1.0")
app.include_router(mcp_server.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "CiNii MCP Server is running"}


def main() -> None:
    uvicorn.run("cinii_mcp_server.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
