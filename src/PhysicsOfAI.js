import { useEffect, useRef, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

// Chapter markdown lives in src/physics-of-ai/; webpack turns each import into a URL we fetch.
const mdFiles = require.context('./physics-of-ai', false, /\.md$/);
const FIG_DIR = process.env.PUBLIC_URL + '/images/physics-of-ai/';

const parts = [
  { name: 'Prologue', chapters: [['01', 'prologue', 'Prologue: Weights and Biases']] },
  {
    name: 'Part I · The Gears',
    chapters: [
      ['02', 'one-neuron', 'One Neuron'],
      ['03', 'world-is-numbers', 'The World Is Numbers'],
      ['04', 'why-scale-matters', 'Why Scale Matters'],
      ['05', 'tensor-operations-as-physics', 'Tensor Operations as Physics'],
      ['06', 'the-collapse', 'The Collapse'],
      ['07', 'uncrumpling-paper', 'Uncrumpling Paper'],
      ['08', 'measuring-wrongness', 'Measuring Wrongness'],
      ['09', 'rolling-downhill', 'Rolling Downhill'],
      ['10', 'sgd', 'SGD: Learning from a Handful at a Time'],
      ['11', 'backprop', 'The Chain Rule Is Backprop'],
      ['12', 'build-it', 'Build It'],
    ],
  },
  {
    name: 'Part II · Making It Learn',
    chapters: [
      ['13', 'memorising-vs-learning', 'Memorising vs Learning'],
      ['14', 'holding-the-network-back', 'Holding the Network Back'],
      ['15', 'starting-right', 'Starting Right'],
      ['16', 'better-steps', 'Better Steps'],
      ['17', 'keeping-things-balanced', 'Keeping Things Balanced'],
      ['18', 'the-recipe', 'The Recipe'],
    ],
  },
  {
    name: 'Part III · Why Depth?',
    chapters: [
      ['19', 'any-function-at-all', 'Any Function At All'],
      ['20', 'when-deeper-gets-worse', 'When Deeper Gets Worse'],
      ['21', 'the-modern-block', 'The Modern Block'],
    ],
  },
  {
    name: 'Part IV · Seeing',
    chapters: [
      ['22', 'looking-through-a-small-window', 'Looking Through a Small Window'],
      ['23', 'zooming-out', 'Zooming Out'],
      ['24', 'real-pictures', 'Real Pictures'],
      ['25', 'what-the-network-sees', 'What the Network Sees'],
    ],
  },
  {
    name: 'Part V · Remembering',
    chapters: [
      ['26', 'text-as-numbers', 'Text as Numbers'],
      ['27', 'words-as-points', 'Words as Points in Space'],
      ['28', 'reading-in-order', 'Reading in Order'],
      ['29', 'gates-and-highways', 'Gates and Highways'],
      ['30', 'the-bottleneck', 'The Bottleneck'],
    ],
  },
  {
    name: 'Part VI · Attention',
    chapters: [
      ['31', 'looking-back', 'Looking Back'],
      ['32', 'self-attention', 'Self-Attention'],
      ['33', 'many-heads-and-positions', 'Many Heads and Positions'],
      ['34', 'the-transformer-block', 'The Transformer Block'],
      ['35', 'attention-is-all-you-need', 'Attention Is All You Need'],
    ],
  },
];

const chapters = parts.flatMap((p) =>
  p.chapters.map(([num, slug, title]) => ({ num, slug, title, part: p.name }))
);

const mdComponents = {
  img: ({ src, alt }) => (
    <figure className="post-figure">
      <img src={src.replace(/^figures\//, FIG_DIR)} alt={alt} loading="lazy" />
      {alt && <figcaption>{alt}</figcaption>}
    </figure>
  ),
  // react-markdown wraps a lone image in <p>; a <figure> can't live inside one
  p: ({ node, children }) =>
    node.children.length === 1 && node.children[0].tagName === 'img' ? children : <p>{children}</p>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="poa-table-wrap">
      <table>{children}</table>
    </div>
  ),
};

function Sidebar({ current }) {
  const [open, setOpen] = useState(false);
  const listRef = useRef(null);

  // keep the current chapter visible in the (independently scrolling) sidebar
  useEffect(() => {
    setOpen(false);
    const list = listRef.current;
    const active = list && list.querySelector('.active');
    if (active && list.scrollHeight > list.clientHeight) {
      list.scrollTop = active.offsetTop - list.clientHeight / 3;
    }
  }, [current]);

  return (
    <nav className={'poa-sidebar' + (open ? ' open' : '')}>
      <button type="button" className="poa-sidebar-toggle" onClick={() => setOpen(!open)}>
        {open ? 'Hide chapters' : 'All chapters'}
      </button>
      <div className="poa-sidebar-list" ref={listRef}>
        <p className="poa-sidebar-title">
          <Link to="/physics-of-ai">The Physics of AI</Link>
        </p>
        {parts.map((p) => (
          <div key={p.name}>
            <p className="poa-sidebar-part">{p.name}</p>
            <ol>
              {p.chapters.map(([num, slug, title]) => (
                <li key={slug}>
                  <Link
                    to={`/physics-of-ai/${slug}`}
                    className={slug === current ? 'active' : undefined}
                  >
                    <span className="poa-num">{Number(num)}</span>
                    {title}
                  </Link>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>
    </nav>
  );
}

function Chapter({ index }) {
  const ch = chapters[index];
  const prev = chapters[index - 1];
  const next = chapters[index + 1];
  const [md, setMd] = useState(null);

  useEffect(() => {
    let live = true;
    setMd(null);
    window.scrollTo(0, 0);
    fetch(mdFiles(`./physics-of-ai_${ch.num}_${ch.slug}.md`))
      .then((r) => r.text())
      // the first line is the series banner ("# Physics of AI — Part …"); we render our own
      .then((text) => live && setMd(text.replace(/^# .*\n/, '')))
      .catch(() => live && setMd('*Could not load this chapter.*'));
    return () => {
      live = false;
    };
  }, [ch]);

  const pager = (
    <nav className="poa-pager">
      {prev ? <Link to={`/physics-of-ai/${prev.slug}`}>← {prev.title}</Link> : <span />}
      {next ? <Link to={`/physics-of-ai/${next.slug}`}>{next.title} →</Link> : <span />}
    </nav>
  );

  return (
    <div className="poa-layout">
      <Sidebar current={ch.slug} />
      <div className="poa-main">
        <p className="post-date">{ch.part}</p>

        <article className="post poa-body">
          {md === null ? (
            <p className="post-date">Loading…</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={mdComponents}
            >
              {md}
            </ReactMarkdown>
          )}
        </article>

        <footer>{pager}</footer>
      </div>
    </div>
  );
}

function PhysicsOfAI() {
  const { slug } = useParams();
  // the series opens straight into the prologue; the sidebar is the table of contents
  if (!slug) return <Navigate to="/physics-of-ai/prologue" replace />;

  const index = chapters.findIndex((c) => c.slug === slug);
  if (index === -1) {
    return (
      <div className="container">
        <p className="back-link">
          <Link to="/physics-of-ai">← The Physics of AI</Link>
        </p>
        <p>Chapter not found.</p>
      </div>
    );
  }
  return <Chapter index={index} />;
}

export default PhysicsOfAI;
