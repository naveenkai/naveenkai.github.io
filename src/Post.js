import { Link, useParams } from 'react-router-dom';
import { getPost } from './writing/posts';

const formatDate = (iso) => {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
};

function Post() {
  const { slug } = useParams();
  const post = getPost(slug);

  if (!post) {
    return (
      <div className="container">
        <p className="back-link"><Link to="/writing">← Back to Writing</Link></p>
        <p>Post not found.</p>
      </div>
    );
  }

  const Body = post.body;

  return (
    <div className="container">
      <p className="back-link"><Link to="/writing">← Back to Writing</Link></p>

      <article className="post">
        <header className="post-header">
          {post.date && <p className="post-date">{formatDate(post.date)}</p>}
          <h1 className="post-title">{post.title}</h1>
          {post.subtitle && <p className="post-subtitle">{post.subtitle}</p>}
        </header>

        <div className="post-body">
          <Body />
        </div>
      </article>

      <footer>
        <p><Link to="/writing">← Back to Writing</Link></p>
      </footer>
    </div>
  );
}

export default Post;
