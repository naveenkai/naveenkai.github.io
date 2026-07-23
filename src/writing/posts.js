import React from 'react';

// Add new posts to the top of this array.
const posts = [
  {
    slug: 'building-effire',
    title: 'My experience building effire.com',
    subtitle:
      'An agent that conducts resume screening, phone screening, AI video interview, and AI coding interview. Built for complete control and transparency.',
    date: '2026-05-07',
    blurb:
      'Five things I lean on now to build reliable agents on top of a primitive that is, at its heart, random.',
    body: () => (
      <>
        <h3>A bit of context</h3>
        <p>
          Around this time last year, I built a project called Interview Agent — an agent that conducts
          interviews given a resume and a job description. It went viral. 5M+ views on Instagram, 136k
          likes, ~12k followers. It was a crazy experience.
        </p>
        <figure className="post-figure">
          <img
            src={process.env.PUBLIC_URL + '/images/interview-agent-reels.jpeg'}
            alt="Two Instagram reels of Interview Agent — the coding round demo and the AI interview demo."
          />
          <figcaption>Interview Agent on Instagram — V2 (left) and V1 (right).</figcaption>
        </figure>
        <p>
          And then I started working on it further. Because the thing I had shipped was maybe 5% of what
          I actually had to build. I had to build voice infrastructure from scratch, an orchestrator,
          telephony and video streaming infra, an eval system, a robust scoring system with minimal error
          rate (+ or -5), and a context-aware agent that could actually carry a real interview.
        </p>
        <p>
          Somewhere in that grind, I got irritated.
        </p>
        <p>
          Irritated with the non-deterministic nature of LLMs. The reason I got into AI in the first
          place was because of how deterministic traditional ML felt — it instilled deterministic
          thinking in me. Now I was working with agents, and that determinism was gone.
        </p>
        <p>
          Don't get me wrong — LLMs are a genuine breakthrough. Language is essential to being human.
          It's how we communicate, reason, and get things done. So a model that works in language is a
          breakthrough because language matters to us.
        </p>
        <p>
          But by their nature, LLMs are random. Ask the same question across two chats and you'll often
          get two different answers — especially for tasks like scoring or writing remarks. The output
          depends heavily on the terminologies you use, and the probability those words showed up in the
          training data plays a huge role in the quality of what comes back. They're vulnerable to
          prompt injection. They carry biases toward whichever ideas dominate the internet.
        </p>
        <p>
          So how do you build something reliable on top of something that random?
        </p>
        <p>
          Over the last year, I've come to think about it through five things. This post is about those
          five.
        </p>

        <h3>1. Context Managers</h3>
        <p>This is the first half of every agent I build.</p>
        <p>
          LLMs, the way I see them now, are good context managers. That's their actual job in an agent
          system. They take in messy context — user input, tool outputs, prior conversation,
          instructions — and they manage it. They route. They reason about it in language. They produce
          the next move.
        </p>
        <p>That's it. That's the role.</p>
        <p>
          The reason this framing helps me is that it stops me from asking LLMs to do things they're bad
          at, like deterministic computation, exact scoring, or strict rule following. Those are not
          context-management problems.
        </p>
        <p>
          If you think of LLMs as the part of your system that automates the repeated thoughts a human
          would have while looking at incoming context — "okay, what is this user asking, what tool
          should I call, what should I say back" — they become a lot less mysterious.
        </p>

        <h3>2. Controlled Entities</h3>
        <p>This is the other half. The deterministic half.</p>
        <p>
          Tools. APIs. Functions. Gmail MCP. A scoring function. A SQL query. A retrieval call. Anything
          that takes defined input parameters and returns defined output parameters.
        </p>
        <p>
          These are your controlled entities. They're predictable. They don't hallucinate. They don't
          drift between runs.
        </p>
        <p>
          The trick to building reliable agents, I've found, is moving as much of the real work as
          possible out of the context manager (the LLM) and into controlled entities (your tools). The
          LLM decides what to do; the tool actually does it.
        </p>
        <p>
          In Interview Agent, the scoring math is a function. The LLM doesn't pick a number out of the
          air — it provides reasoning, the tool computes the score.
        </p>
        <p>
          This split — context managers + controlled entities — is, I think, the single most useful
          mental model I have for agent design.
        </p>

        <h3>3. Context Engineering</h3>
        <p>
          Now, the randomness doesn't fully go away just because you've offloaded computation to tools.
          The context manager is still bounded by two things: the context it gets, and the format it's
          given in.
        </p>
        <p>That's where context engineering comes in.</p>
        <p>Context engineering is the craft of shaping what the agent sees. It's deciding:</p>
        <ul>
          <li>Should I extract specific fields and pass them in as JSON instead of dumping the raw text?</li>
          <li>Can I compress this long history without losing the bits that matter?</li>
          <li>When should I inject new context mid-conversation?</li>
        </ul>
        <p>That last one matters more than people realize. A few examples from my own work:</p>
        <ul>
          <li>
            When a user clicks a button in the UI, I inject a short context note so the agent knows the
            click happened.
          </li>
          <li>
            When a user submits code, I run a separate API call to evaluate the code, then inject that
            evaluation summary back into the agent's context. The agent now has a real, structured
            signal to talk about — not a guess.
          </li>
        </ul>
        <p>
          This becomes critical in voice agents, where you need the conversation to feel meaningful,
          real-time, and contextually aware. The latency budget is brutal, and the agent needs the right
          context at the right moment, in the right format. Not all of it. Not raw. Engineered.
        </p>

        <h3>4. Eval</h3>
        <p>
          If I had to pick the single most important thing in building any agent, it's eval.
        </p>
        <p>Your evaluation dataset and the metrics you track are everything.</p>
        <ul>
          <li>Want to swap in a cheaper model to cut cost? You need eval to know if it actually still works.</li>
          <li>Want to try a new prompt or a new context-engineering trick? You need eval to know if it's better, worse, or just different.</li>
          <li>Want to add a new tool, change an architecture, refactor an orchestrator? Eval tells you whether the change moved the needle.</li>
        </ul>
        <p>
          Without eval, every "improvement" is vibes. With eval, you have an internal benchmark — and
          benchmarks are what let you make changes confidently instead of nervously.
        </p>
        <p>
          I'll be honest: building a good eval set is unglamorous. It's slow, manual, and feels like a
          detour from the "real" work. It is the real work. Skip it and you'll spend ten times longer
          debugging things you can't measure.
        </p>

        <h3>5. Transparency, Trust, Control</h3>
        <p>
          This is the part I care about the most. Probably because it forces me to stay first-principled
          and realistic about what I'm building.
        </p>
        <p>
          I think of an agent as an excited intern who inherited their weights from the internet. It's
          smart. It's capable. It's also, sometimes, foolish — depending on the model, the context, and
          what you're asking it to do.
        </p>
        <p>
          When you work with an intern like that, you don't blindly trust them. You ask them to show
          their work.
        </p>
        <p>So I hold my agents to the same standard:</p>
        <ul>
          <li>When the agent says something, can it cite evidence?</li>
          <li>When it scores a candidate, is it explaining why it scored that way?</li>
          <li>When it makes a decision, can a human follow the trail?</li>
        </ul>
        <p>That's transparency.</p>
        <p>
          Trust comes after that, and trust is earned narrowly. I trust a specific agent on a specific
          task within a specific threshold. Below the threshold, the work goes to a human. The scoring
          system isn't there to replace the human reviewer — it's there to know when to call them in.
        </p>
        <p>And finally, control.</p>
        <p>
          Humans should set the rules. Humans should be able to stop the system within five minutes. Not
          five days, not after a long incident review — five minutes. If you can't kill it fast, you
          don't really control it.
        </p>
        <p>
          The more autonomous your agent, the more important this gets. An agent that acts in the world
          — sends emails, books interviews, triggers workflows — is an agent that can also send the
          wrong email, book the wrong interview, trigger the wrong workflow. The kill switch isn't a
          nice-to-have. It's the price of letting the thing run at all.
        </p>

        <p>
          None of these are revolutionary on their own. But put together, they're the lens I use now
          when I look at any agent system — mine or someone else's. They're how I stay sane in a field
          where the underlying primitive is, at its heart, random.
        </p>
        <p>
          If you're building agents and any of this resonates, I'd love to hear.
        </p>
        <p>— Naveen</p>
      </>
    ),
  },
  {
    slug: 'early-20s',
    title: 'Early 20s',
    blurb: 'On working harder than people say you should — and knowing where to stop.',
    body: () => (
      <>
        <p>
          People say you shouldn't work all the time in your early 20s. Which is true. But I feel you
          should work harder than people think you should. We don't understand how unfair life is. We
          are born with different starting lines in an immature society. Sometimes you get lucky, but
          most of the time you don't.
        </p>
        <p>
          Working hard maximizes your chances of success. It compounds over time. You get better
          opportunities and leverage from your hard work.
        </p>
        <p>
          Don't just jump in, to take some random path, and end up spending a lot of time there. It is
          really hard to unlearn and come back from that to explore other paths. And a lot of time
          drain! We should spend enough time making big decisions, like choosing the people and the
          work we want to work on.
        </p>
        <p>You should definitely respect patterns sometimes!</p>
        <p>
          But it is a bad idea to go too hard on yourself. It's not what you achieve, but what you
          manage to achieve with what you have. You should keep that in check and allow yourself to
          enjoy.
        </p>
        <p>But work hard to your max if you can't afford to depend on luck.</p>
        <p>— Naveen K</p>
      </>
    ),
  },
  {
    slug: 'why-1d-advice-wont-work',
    title: "Why 1D Advice Won't Work",
    blurb: 'Generic advice fails because it collapses a multi-dimensional life into a single axis.',
    body: () => (
      <>
        <p>1D advice is everywhere. It's generic, shallow, and often meaningless.</p>
        <p>
          The internet is flooded with it. Some random person shows up and throws out some random line
          of advice. It has nothing to do with your life. They say things like "xyz people are like
          this" or "do xyz things to get better at something." A few might be useful, but the
          percentage of advice that actually helps is less than 5%. Yet people still spend their energy
          thinking about it.
        </p>
        <p>
          Even in real conversations, when someone gives you straight-up advice without understanding
          your situation in more than one dimension. It is meaningless but seen as a sign of confidence
          when you give a quick answer/advice. I don't know but being so certain is seen as confidence.
          But real certainty takes time. It comes from seeing multiple dimensions, not just one.
        </p>
        <p>
          I suspect this one-dimensional way of thinking is part of why people today feel lonely,
          confused, and disconnected. When you rush to a conclusion without exploring the different
          dimensions, you either trap yourself with 1D thoughts about your own life, or you let someone
          else impose their 1D thoughts on you. Both lead to shallow understanding and poor decisions.
        </p>
        <p>
          And that's why I feel if it is not professional problem it is mostly YOU who can figure it
          out by understanding different dimensions of your own self in a particular problem.
          First-principles is again is a great framework. A lot of times Instincts.
        </p>
        <p>
          I think Experts are people who understand multiple dimensions of a field. And that kind of
          understanding only comes with time by spending years exploring, failing, learning, and going
          deeper.
        </p>
        <p>
          This is also why I don't believe experts will be replaced in a post-AGI world. AGI learns
          from data produced by us, which means its knowledge is still constrained by the patterns it
          finds. There will always be room for nuance, for deeper perspectives, for debates that
          require someone who has lived those dimensions in practice. That's where human experts will
          remain irreplaceable.
        </p>
        <p>— Naveen K</p>
      </>
    ),
  },
  {
    slug: 'its-easy-to-get-stuck',
    title: 'It is very easy to get stuck',
    blurb: "A lot of movement isn't the same as moving forward.",
    body: () => (
      <>
        <p>I think it's very easy to get stuck in life in today's time.</p>
        <p>
          Spending a decade being incredibly busy and stressed creating a ton of movement but not
          moving forward.
        </p>
        <p>
          Or getting stuck in things that are urgent but unimportant, getting caught up in power and
          status games, or getting trapped in FOMO and trend loops is so easy.
        </p>
        <p>
          It feels like you're getting important things done, doing something cool and fun. But it's
          clearly a trap. It is a ton of movement without actually moving forward.
        </p>
        <p>
          We chase validation instead of clarity. The world rewards noise, not depth — and that's where
          it gets dangerous.
        </p>
        <p>This slows down your potential.</p>
        <p>Cutting off the unnecessary stuff and following the top line is really important.</p>
        <p>— Naveen K</p>
      </>
    ),
  },
  {
    slug: 'keep-going',
    title: "You don't know what you don't know. Keep going!",
    blurb: "When you feel stuck, remember: you don't yet know what time will teach you.",
    body: () => (
      <>
        <p>
          Whenever you feel stuck, not seeing hope anywhere close, when you get suicidal thoughts, when
          you feel you deserve better but you're not getting it, when you're thinking about your
          insecurities, when you're questioning your capabilities, or when others are questioning your
          capabilities—and amidst all this chaos—just remember, everything is going to be okay with
          time.
        </p>
        <p>So what exactly happens with time?</p>
        <p>You gain knowledge about the TRUTH. Because you don't know enough about things that matter!</p>
        <ul>
          <li>
            A new person may come into your life and can completely change your perspective about
            yourself and your flaws.
          </li>
          <li>An inspirational story you read may change the way you think about the world.</li>
          <li>Your icon may share their experience in searching for this TRUTH.</li>
        </ul>
        <p>All good pieces of words, stories, and works make you feel alive again.</p>
        <p>
          Make sure you're heading in the right direction—you'll learn about the TRUTH along the way.
        </p>
        <p>
          You'll laugh at all the unnecessary things you once worried about, realizing they were far
          from the TRUTH when you look back.
        </p>
        <p>I wish you look back and laugh :)</p>
        <p>— Naveen K</p>
      </>
    ),
  },
  {
    slug: 'biases',
    title: 'Experience, help everyone experience: Biases',
    blurb: 'From LUCA to first principles — why our inherited biases are baggage we can shed.',
    body: () => (
      <>
        <p>
          LUCA – The Last Universal Common Ancestor. The single ancestral cell from which all three
          domains of life emerged—Bacteria, Archaea, and Eukarya.
        </p>
        <p>Humans, animals, plants, methanogens, cocci—We come from a single ancestor.</p>
        <p>
          For 3.6 billion years, life has evolved, and along the way, we have carried a baggage called
          biases.
        </p>
        <p>Well what is a bias? — An axis that doesn't contribute in this multi-dimensional world.</p>
        <p>
          These invisible weights, born of survival instincts, cultural echoes, and societal critiques,
          have become humanity's collective baggage. They linger in how we judge, exclude, and
          limit—ourselves and others.
        </p>
        <p>
          We are here for just a short interval. What truly matters in this fleeting moment? Not
          hurting animals. Not destroying nature. And most importantly, creating value—something that
          helps humanity live better, rather than reinforcing biases, as most social media algorithms
          tend to do.
        </p>
        <p>
          I like to think that this is not a zero sum game where we can use biases as weapons to be
          superior. It's a positive sum game, or should be a positive sum game where everyone can win by
          experiencing it to its maximum.
        </p>
        <h3>Seeing Through First Principles</h3>
        <p>
          First-principles thinking allows us to dissect and dismantle biases. To look at the world not
          through the lens of inherited assumptions but through fundamental truths. This is how biases
          can be rectified, restructured, and ultimately erased.
        </p>
        <p>
          Find places where bias is minimal. Surround yourself with people who question biases rather
          than reinforce them. Sometimes, even if you can't escape bias immediately, introduce an axis
          shift—be delusional enough, see through first principles stand point—until you can access
          environments that foster unbiased growth.
        </p>
        <p>
          Let's make sure the baggage we've carried doesn't hurt the experience— yours or anyone else's.
        </p>
        <p>— Naveen K</p>
      </>
    ),
  },
  {
    slug: 'why-life-seems-complicated',
    title: 'Why Life Seems Complicated When We Grow Up? A Multi-dimensional Journey',
    blurb: "Growing up isn't getting taller; it's your world gaining dimensions.",
    body: () => (
      <>
        <p>
          Remember when you were a kid? Life felt like a straight road. Your biggest concerns were nap
          time, snack breaks, and whether the swings were free at recess. Happiness was simple—a ice
          cream cone, a new toy, or a day at the park. But as we grow older, that simplicity fades.
          Suddenly, life feels like a tangled knot of responsibilities, relationships, and existential
          questions. Why does adulthood feel so complicated? Let's unpack this using dimensions.
        </p>
        <h3>Childhood in 2D: Happiness vs. Time</h3>
        <p>
          Imagine your life as a graph when you're born. The X-axis is time, and the Y-axis is
          happiness. Your daily "problems" fit neatly into this 2D space. Hungry? Cry. Bored? Play. Sad?
          Hug. Solutions are straightforward, and your "feasible region"—the sweet spot where needs are
          met—is easy to find. Life is linear, predictable, and wonderfully simple.
        </p>
        <p>But then,</p>
        <h3>Adulthood: When Life Gains Dimensions</h3>
        <p>
          Growing up isn't just about getting taller—it's about your world expanding into
          multidimensional space. Suddenly, new axes pop up:
        </p>
        <ul>
          <li>Career</li>
          <li>Relationships</li>
          <li>Finances</li>
          <li>Health</li>
          <li>Purpose</li>
          <li>Social Expectations</li>
        </ul>
        <p>
          Each responsibility, dream, or crisis adds another dimension. Your 2D graph becomes a 3D
          cube, then a 4D hypercube, and soon you're lost in a labyrinth of choices. Want to chase a
          dream job? That affects your finances, time, and relationships. Prioritize family? That might
          reshape your career trajectory. Every decision ripples across dimensions, and the "right"
          answer is rarely obvious.
        </p>
        <h3>The Feasible Region: Life's Optimization Game</h3>
        <p>
          In math, a feasible region is where all constraints overlap—a zone where solutions exist. As
          kids, our feasible region was a spacious bubble: eat, sleep, play, repeat. But adulthood?
          It's like playing Jenga in a hurricane. Every new dimension (a mortgage, a sick parent, a
          career pivot) tightens the constraints. The feasible region shrinks, and balancing it all
          feels impossible.
        </p>
        <p>
          Here's the kicker: hardship accelerates complexity. When you hit a low point—a breakup, job
          loss, or loss of a loved one—it's like adding a new dimension overnight. Suddenly, you're
          graphing grief or fear alongside everything else. The more you experience, the more axes you
          juggle.
        </p>
        <h3>Why Can't We Go Back to 2D?</h3>
        <p>
          We can't unlearn or un-live. Once you've known love, loss, or ambition, those dimensions
          stick. Nostalgia for simplicity is natural, but it's like trying to flatten a cube back into a
          square—you'd lose depth, meaning, and growth. The complexity isn't a flaw; it's proof you're
          living.
        </p>
        <h3>The Beauty of the Mess</h3>
        <p>
          Yes, adulthood is complicated. But in that complexity lies richness. Those dimensions—career,
          love, growth, even pain—are what make life dimensional. The feasible region isn't a tiny dot;
          it's a shifting, evolving space where you learn, adapt, and thrive.
        </p>
        <p>
          So next time life feels like a Rubik's Cube you'll never solve, remember: you're not meant to.
          You're meant to keep twisting, exploring, and discovering new corners of your multidimensional
          universe. And honestly? That's kind of magical.
        </p>
        <p>— Naveen K</p>
      </>
    ),
  },
];

export default posts;
export const getPost = (slug) => posts.find((p) => p.slug === slug);
