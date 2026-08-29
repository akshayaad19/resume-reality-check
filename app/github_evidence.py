"""GitHub-backed evidence: turns a user's public repos into evidence chunks.

Fetches recently updated public repos for a GitHub username, and for each one
builds an evidence chunk (README + primary language + verified dependencies +
structural signals) in the same plain-text-bullet format as resume evidence,
so it can be merged into the same EvidenceIndex and retrieved alongside
resume bullets.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import List, Optional

import requests

GITHUB_API_BASE = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 10
REPO_LIMIT = 10
README_MAX_CHARS = 2000
VERIFIED_DEPS_MAX_PACKAGES = 40


class GitHubFetchError(Exception):
    """Raised when GitHub evidence can't be fetched (bad username, rate limit,
    network failure, etc). Callers should catch this and fall back to
    resume-only evidence rather than letting it propagate."""


def _get_session() -> requests.Session:
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def _list_recent_repos(session: requests.Session, username: str) -> List[dict]:
    try:
        response = session.get(
            f"{GITHUB_API_BASE}/users/{username}/repos",
            params={"sort": "updated", "direction": "desc", "per_page": REPO_LIMIT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise GitHubFetchError(f"Failed to reach GitHub API: {e}") from e

    if response.status_code == 404:
        raise GitHubFetchError(f"GitHub user {username!r} not found")
    if response.status_code in (403, 429):
        raise GitHubFetchError("GitHub API rate limit exceeded")
    if not response.ok:
        raise GitHubFetchError(
            f"GitHub API returned {response.status_code} listing repos for {username!r}"
        )
    return response.json()[:REPO_LIMIT]


def _decode_base64_file(data: dict) -> Optional[str]:
    if not isinstance(data, dict) or data.get("encoding") != "base64" or not data.get("content"):
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return None


def _fetch_readme_text(session: requests.Session, owner: str, repo: str) -> Optional[str]:
    try:
        response = session.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None
    if not response.ok:
        return None
    text = _decode_base64_file(response.json())
    return text.strip() if text else None


def _fetch_file_text(session: requests.Session, owner: str, repo: str, path: str) -> Optional[str]:
    try:
        response = session.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None
    if not response.ok:
        return None
    return _decode_base64_file(response.json())


def _parse_requirements_txt(text: str) -> List[str]:
    """Extract package names from a requirements.txt, stripping comments,
    version pins, extras, and environment markers, e.g.
    "sentence-transformers==2.2.2" -> "sentence-transformers" and
    "uvicorn[standard]>=0.20; python_version>='3.8'" -> "uvicorn"."""
    packages = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http://", "https://")):
            continue
        name = line.split(";", 1)[0]
        name = re.split(r"[\[=<>!~\s]", name, 1)[0].strip()
        if name:
            packages.append(name)
    return packages


def _parse_package_json(text: str) -> List[str]:
    """Extract dependency package names from package.json's "dependencies"
    and "devDependencies" objects."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    packages = []
    for key in ("dependencies", "devDependencies"):
        deps = data.get(key) if isinstance(data, dict) else None
        if isinstance(deps, dict):
            packages.extend(deps.keys())
    return packages


def _fetch_top_level_contents(session: requests.Session, owner: str, repo: str) -> List[dict]:
    try:
        response = session.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return []
    if not response.ok:
        return []
    data = response.json()
    return data if isinstance(data, list) else []


def _has_ci_workflows(session: requests.Session, owner: str, repo: str, top_level: List[dict]) -> bool:
    """A `.github/workflows/` folder is one level below the top-level listing,
    so only make the extra request when a `.github` dir is actually there."""
    has_github_dir = any(
        entry.get("type") == "dir" and entry.get("name") == ".github" for entry in top_level
    )
    if not has_github_dir:
        return False
    try:
        response = session.get(
            f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/.github",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return False
    if not response.ok:
        return False
    github_dir_contents = response.json()
    if not isinstance(github_dir_contents, list):
        return False
    return any(
        entry.get("type") == "dir" and entry.get("name") == "workflows"
        for entry in github_dir_contents
    )


def _detect_structural_signals(
    session: requests.Session, owner: str, repo: str, top_level: List[dict]
) -> List[str]:
    names_by_type = {
        (entry.get("name"), entry.get("type")) for entry in top_level
    }

    signals = []
    if ("Dockerfile", "file") in names_by_type:
        signals.append("Dockerfile present")
    if ("k8s", "dir") in names_by_type or ("helm", "dir") in names_by_type:
        signals.append("k8s/helm config present")
    if _has_ci_workflows(session, owner, repo, top_level):
        signals.append("CI workflow present")
    return signals


def _detect_verified_dependencies(
    session: requests.Session, owner: str, repo: str, top_level: List[dict]
) -> List[str]:
    """Fetch and parse requirements.txt / package.json (when present at the
    repo's top level) into their actual listed package names.

    Unlike a boolean "file present" signal - which is too generic to confirm
    any specific skill, since almost every Python/JS repo has one - the
    individual package names inside these files are typically machine-
    generated from what's actually installed (e.g. via `pip freeze`), making
    them hard to fake and strong evidence for whatever skill a listed package
    actually implements."""
    names_by_type = {(entry.get("name"), entry.get("type")) for entry in top_level}
    lines = []

    if ("requirements.txt", "file") in names_by_type:
        text = _fetch_file_text(session, owner, repo, "requirements.txt")
        packages = _parse_requirements_txt(text)[:VERIFIED_DEPS_MAX_PACKAGES] if text else []
        if packages:
            lines.append(f"Verified dependencies (from requirements.txt): {', '.join(packages)}")

    if ("package.json", "file") in names_by_type:
        text = _fetch_file_text(session, owner, repo, "package.json")
        packages = _parse_package_json(text)[:VERIFIED_DEPS_MAX_PACKAGES] if text else []
        if packages:
            lines.append(f"Verified dependencies (from package.json): {', '.join(packages)}")

    return lines


def _build_repo_evidence_chunk(session: requests.Session, owner: str, repo: dict) -> Optional[str]:
    repo_name = repo.get("name")
    if not repo_name:
        return None

    language = repo.get("language") or "Unknown"
    readme_text = _fetch_readme_text(session, owner, repo_name)
    top_level = _fetch_top_level_contents(session, owner, repo_name)
    signals = _detect_structural_signals(session, owner, repo_name, top_level)
    verified_deps = _detect_verified_dependencies(session, owner, repo_name, top_level)

    signals_summary = (
        f"Structural signals found: {', '.join(signals)}"
        if signals
        else "Structural signals found: none"
    )

    parts = [f"Repo: {repo_name}", f"Primary language: {language}"]
    if readme_text:
        parts.append(f"README: {readme_text[:README_MAX_CHARS]}")
    parts.extend(verified_deps)
    parts.append(signals_summary)
    return "\n".join(parts)


def fetch_github_evidence(username: str) -> List[str]:
    """Fetch evidence chunks from a GitHub user's public repos.

    Looks at the REPO_LIMIT most recently updated public repos and builds one
    evidence chunk per repo (README + primary language + structural signals),
    in the same plain-text-bullet format as resume evidence bullets.

    Raises GitHubFetchError if the username doesn't exist, the API call fails,
    or the request is rate-limited - callers should catch this and fall back
    to resume-only evidence rather than crashing.
    """
    session = _get_session()
    repos = _list_recent_repos(session, username)

    evidence_chunks = []
    for repo in repos:
        chunk = _build_repo_evidence_chunk(session, username, repo)
        if chunk:
            evidence_chunks.append(chunk)
    return evidence_chunks
