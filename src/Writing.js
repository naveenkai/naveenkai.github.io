import { Link } from 'react-router-dom';
import posts from './writing/posts';

const formatDate = (iso) => {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
};

function Writing() {
  return (
    <div className="container">
      <header className="writing-header">
        <p className="back-link"><Link to="/">← Back</Link></p>
        <h1>Writing</h1>
      </header>

      <section>
        {posts.map((p) => (
          <div className="entry post-preview" key={p.slug}>
            {p.date && <p className="post-date">{formatDate(p.date)}</p>}
            <p>
              <Link to={`/writing/${p.slug}`} className="post-title-link">
                <strong>{p.title}</strong>
              </Link>
            </p>
            {p.subtitle && <p className="post-subtitle">{p.subtitle}</p>}
            {p.blurb && <p className="post-blurb">{p.blurb}</p>}
          </div>
        ))}
      </section>
    </div>
  );
}

export default Writing;
