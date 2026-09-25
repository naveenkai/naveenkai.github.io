import { useEffect, useLayoutEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';

// Width / height as the browser displays each photo (EXIF rotation applied)
const PORTRAIT = [7, 11, 15, 17, 19, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39,
  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56];
const RATIO = { 1: 1, 2: 1, 37: 923 / 1280, 55: 1075 / 1280 };
PORTRAIT.forEach((n) => { RATIO[n] = 3 / 4; });

// Hampi first, then everything else in folder order
const HAMPI = Array.from({ length: 18 }, (_, i) => 39 + i);
const ORDER = [...HAMPI, ...Array.from({ length: 38 }, (_, i) => i + 1)];

const GAP = 6;

// Each block is a set of side-by-side columns; the number is how many photos stack in that column.
// Blocks cycle in this order, so big single photos sit next to stacks of smaller ones.
const WIDE_BLOCKS = [[1, 2], [1, 1, 1, 1], [2, 1, 1], [1, 1, 1], [1, 2, 1], [2, 1], [1, 1, 1, 1, 1], [1, 1, 2]];
const NARROW_BLOCKS = [[1, 2], [1, 1], [2, 1], [1], [1, 1, 1]];

const src = (n) => `${process.env.PUBLIC_URL}/images/gallery/${String(n).padStart(2, '0')}.jpg`;

const photos = ORDER.map((n, index) => ({ n, index, ratio: RATIO[n] || 4 / 3 }));

// Height at which every column in a block ends flush, given each photo keeps its aspect ratio.
// A column of n photos with S = sum(1 / ratio) is (H - GAP * (n - 1)) / S wide.
const blockHeight = (cols, width) => {
  let inv = 0;
  let extra = 0;
  cols.forEach((col) => {
    const s = col.reduce((sum, p) => sum + 1 / p.ratio, 0);
    inv += 1 / s;
    extra += (GAP * (col.length - 1)) / s;
  });
  return (width - GAP * (cols.length - 1) + extra) / inv;
};

// Absolute { x, y, w, h } for every photo, so nothing is ever cropped or stretched
function layout(width) {
  const narrow = width < 700;
  const blocks = narrow ? NARROW_BLOCKS : WIDE_BLOCKS;
  const maxH = width * (narrow ? 1.4 : 0.5);
  const tiles = [];
  let y = 0;
  let i = 0;

  for (let b = 0; i < photos.length; b++) {
    const cols = [];
    for (const count of blocks[b % blocks.length]) {
      const col = photos.slice(i, i + count);
      if (col.length) cols.push(col);
      i += col.length;
    }
    // Too tall (e.g. several portraits landed together): widen with extra single columns
    let h = blockHeight(cols, width);
    while (h > maxH && i < photos.length) {
      cols.push([photos[i++]]);
      h = blockHeight(cols, width);
    }

    let x = 0;
    for (const col of cols) {
      let s = 0;
      for (const p of col) s += 1 / p.ratio;
      const w = (h - GAP * (col.length - 1)) / s;
      let cy = y;
      for (const p of col) {
        const ph = w / p.ratio;
        tiles.push({ ...p, x, y: cy, w, h: ph });
        cy += ph + GAP;
      }
      x += w + GAP;
    }
    y += h + GAP;
  }
  return { tiles, height: Math.max(0, y - GAP) };
}

function Gallery() {
  const [open, setOpen] = useState(null);
  const [width, setWidth] = useState(0);
  const gridRef = useRef(null);

  useLayoutEffect(() => {
    const el = gridRef.current;
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { tiles, height } = useMemo(() => layout(width), [width]);

  const step = useCallback(
    (d) => setOpen((i) => (i === null ? i : (i + d + photos.length) % photos.length)),
    []
  );

  useEffect(() => {
    if (open === null) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(null);
      if (e.key === 'ArrowRight') step(1);
      if (e.key === 'ArrowLeft') step(-1);
    };
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [open, step]);

  return (
    <div className="gallery-page">
      <header className="writing-header">
        <p className="back-link"><Link to="/">← Back</Link></p>
        <h1>Gallery</h1>
      </header>

      <div className="gallery-grid" ref={gridRef} style={{ height }}>
        {width > 0 && tiles.map((p) => (
          <button
            type="button"
            key={p.n}
            className="gallery-tile"
            style={{ left: p.x, top: p.y, width: p.w, height: p.h }}
            onClick={() => setOpen(p.index)}
            aria-label={`Open photo ${p.n}`}
          >
            <img src={src(p.n)} alt="" loading="lazy" decoding="async" />
          </button>
        ))}
      </div>

      {open !== null && (
        <div className="lightbox" onClick={() => setOpen(null)} role="dialog" aria-modal="true">
          <button type="button" className="lightbox-nav prev" aria-label="Previous"
            onClick={(e) => { e.stopPropagation(); step(-1); }}>‹</button>
          <img src={src(photos[open].n)} alt="" onClick={(e) => e.stopPropagation()} />
          <button type="button" className="lightbox-nav next" aria-label="Next"
            onClick={(e) => { e.stopPropagation(); step(1); }}>›</button>
        </div>
      )}
    </div>
  );
}

export default Gallery;
