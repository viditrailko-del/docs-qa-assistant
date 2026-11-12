---
title: "Sync vs. Async Culture"
description: "Guidance on when to default to asynchronous communication and when to use synchronous conversation at GitLab."
---
In distributed systems, async and sync aren't ideologies. They're tools, each with a job. Async messaging (queues, pub/sub, event streams) is how you scale. It decouples producers from consumers, absorbs bursty load, and lets services keep working when peers are slow or down. Sync calls (RPC, request/response) are how you coordinate when correctness depends on a fresh, agreed-upon answer right now: a payment authorization, a lock acquisition, a leader election. Build everything async and you get a system that scales beautifully but can't make a decision. Build everything sync and you get a system that's correct but brittle, where one slow node stalls the whole graph.

The same pattern applies to how we work. Slack, docs, and GitLab issues are our async layer. They scale across timezones, create durable artifacts, and let people work without blocking on each other. That's most of the work, and it should be. But some decisions require a synchronous transaction. Aligning on a strategy shift, resolving a real disagreement, building trust with a new exec, debugging a problem where the context isn't yet legible enough to write down, these are the moments where the round-trip latency of async (a thread that takes four days and seventeen replies to converge) is worse than thirty minutes of sync. The failure mode of "async-only" is the same failure mode as an event-driven architecture with no synchronous endpoints. You get eventual consistency on decisions. Threads fork, context drifts, no one knows who decided what, and the work that needed a clean commit point becomes a slow-moving consensus problem instead.

Async is one of GitLab's real strengths, and we're not walking away from it. A durable record of what's going on is a competitive advantage most companies don't have and can't easily build, and it's even more valuable now that agents can use that context. I want to protect it.

The world we're building for now demands that we get better at sync than we've ever been. Speed matters more in the agentic era than it did in the era we built our async culture in. The companies that will win the next decade are the ones that can decide fast, align fast, and ship fast. We built the async muscle. Now we build the sync one. One isn't a substitute for the other.

Building trust with each other and with our teams takes real time together. In a completely async culture, new hires at all levels were telling us they feel lonely when they start. They lack context, they're pointed at things to "go read and figure out," and the human connection that helps a new person actually absorb a company's culture isn't there. That's a problem, and we need more than coffee chats to fix it. In some cases, we've conflated "async-first" with "you're on your own," and that's not what async was ever supposed to mean. We can do better, and I encourage each of us to revisit how we bring new team members along, with more deliberate sync time, mentorship, and the kind of context transfer that genuinely only happens in conversation.

One last thing before I get prescriptive: sync time is not a substitute for prep. If you walk into a meeting without reading the agenda or the pre-read, you're not doing sync, you're just not prepared for the meeting. That's a fair critique and it's on all of us, myself included, to hold the bar there.

So let me make this concrete. Here's what I'd ask all of us to hold ourselves to:

## Async — do

- Default here for status, FYIs, decisions with clear context, proposals, and anything that benefits from being searchable later
- Write the issue, doc, or thread as if the reader has none of your context, because they don't. Use AI tools to make the content as simple, concise, and consumable as possible.
- Tag decision owners explicitly and set a "respond by" date so threads don't drift
- Close the loop in writing once a decision is made, even if it was made in a meeting
- Use documents for content that requires a narrative like proposals or memos. Use GitLab issues for work that has a lifecycle, and use Slack for ephemeral coordination
- Read the full content before replying, use AI tools for summarizations. Async breaks down when people respond to the subject line

## Async — don't

- Don't use async to avoid a hard conversation that needs a real exchange
- Don't let a thread run for a week when a 20-minute call would resolve it
- Don't post a wall of text and expect engagement; if it's that long, it needs a TL;DR and a clear ask
- Don't confuse "I sent a message" with "we aligned"

## Sync — do

- Use it for decisions under ambiguity, conflict resolution, building trust with new people, and brainstorming where the ideas aren't formed enough to write down
- Spend real sync time bringing new team members up to speed. Walk them through the context, introduce them to people, answer their questions in conversation. "Go read the handbook" is a starting point, not an onboarding plan
- Send the agenda and pre-read ahead, with the decision or outcome you're driving toward stated explicitly
- Expect attendees to come prepared; if more than a couple haven't read the pre-read, end the meeting and reschedule
- Capture the decision and owners in writing within 24 hours, in the relevant issue or doc, so the sync output becomes async durable
- Keep the invite list tight; optional means optional

## Sync — don't

- Don't run a meeting to share information that could've been a doc
- Don't use sync as a substitute for thinking; if you haven't formed a view, write first, meet second
- Don't let recurring meetings outlive their purpose. Kill them or shrink them every quarter
- Don't treat showing up as your contribution. Prep is required to participate; contributing to achieving the meeting objective is your contribution
- Remove yourself from meetings where you don't have anything to contribute, and treat others' time as sacred as your own

We built the async muscle that scaled GitLab to where we are. The next chapter requires us to build the sync muscle in balance with async that will let us move at the speed the agentic era demands. A high-performing team, just like high-performing software, has to do both well.
