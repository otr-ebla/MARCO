# Issue tracker: GitHub

Issues and PRDs for this repository live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- Create an issue with `gh issue create --title "..." --body "..."`.
- Read an issue with `gh issue view <number> --comments` and include its labels.
- List issues with `gh issue list`, using state and label filters where appropriate.
- Comment with `gh issue comment <number> --body "..."`.
- Apply or remove labels with `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close an issue with `gh issue close <number> --comment "..."`.

Infer the repository from the current clone's Git remote.

## Skill conventions

When a skill says to publish to the issue tracker, create a GitHub issue. When it says to fetch a relevant ticket, use `gh issue view <number> --comments`.
