import { Link } from 'react-router-dom';
import posts from './writing/posts';

const L = (href, text) => (
  <a href={href} target="_blank" rel="noopener noreferrer">{text}</a>
);

const formatDate = (iso) => {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
};

// Smooth-scroll to a section without touching the URL hash (HashRouter owns it).
const scrollToId = (e, id) => {
  e.preventDefault();
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth' });
};

function Portfolio() {
  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <img
          src={process.env.PUBLIC_URL + '/images/naveenk_dp.jpeg'}
          alt="Naveen K"
          className="avatar"
        />
        <h1>Naveen K</h1>
        <p className="tagline">I build AI systems that see, speak, and reason.</p>
        <p className="links-row">
          <a href="mailto:naveenk.ai.engineer@gmail.com">naveenk.ai.engineer@gmail.com</a>
        </p>
        <p className="links-row">
          {L('https://github.com/naveenkai', 'GitHub')}
          {' | '}
          {L('https://x.com/naveenk_ai', 'X')}
          {' | '}
          {L('https://www.linkedin.com/in/naveenkai/', 'LinkedIn')}
          {' | '}
          {L('https://www.youtube.com/@naveenk_ai', 'YouTube')}
        </p>
        <nav className="section-nav">
          <a href="#about" onClick={(e) => scrollToId(e, 'about')}>About</a>
          <a href="#career" onClick={(e) => scrollToId(e, 'career')}>Career</a>
          <a href="#projects" onClick={(e) => scrollToId(e, 'projects')}>Projects</a>
          <a href="#research" onClick={(e) => scrollToId(e, 'research')}>Research</a>
          <a href="#writing" onClick={(e) => scrollToId(e, 'writing')}>Writing</a>
        </nav>
      </aside>

      {/* Content */}
      <main className="content">
      {/* About */}
      <section id="about">
        <h2>About</h2>
        <p>
          I'm an AI engineer and researcher based in Bengaluru. I work on voice AI agents, computer vision systems,
          and agentic AI pipelines. I started working on AI because I was confused about life — deep math truths,
          stats behind the world, patterns, and some good frameworks to think about life drew me in.
        </p>
        <p>
          I can research and engineer AI — ML, deep learning, generative AI, agentic systems,
          and whatever else it takes to make a system intelligent.
        </p>
      </section>

      {/* Career */}
      <section id="career">
        <h2>Career</h2>

        <div className="entry">
          <p>
            <strong>AI Engineer</strong>, Ekfrazo Technologies, Bengaluru.
            <span className="year">Sep 2025 –</span>
          </p>
          <ul>
            <li>
              Voice AI agents with 95% automation at 500–600ms latency. RAG systems auto-resolving
              70–80% of manual queries. Multi-model reasoning and debate.
            </li>
            <li>Chat with real-time financial data using plain English.</li>
            <li>Automated financial contract analysis using Power Automate.</li>
            <li>
              Built evals and monitoring systems for the agents; made LLMs/agents transparent by
              holding their results accountable.
            </li>
          </ul>
          <p className="stack">
            Stack: Pipecat, LiveKit, LangGraph, LangChain, Exotel, Power Automate, Copilot Studio, GCP.
          </p>
        </div>

        <div className="entry">
          <p>
            <strong>AI/ML Engineer</strong>, Indus Vision, Bengaluru.
            <span className="year">Jun 2024 – Aug 2025</span>
          </p>
          <ul>
            <li>
              Led POSTURA — a 2D clinical gait analysis system. 85–95% keypoint accuracy, 98% gait
              phase detection. Perspective correction and step measurement without 3D sensors.
              {' '}{L('https://youtu.be/_PCpW9ioZr4', 'POSTURA demo')}, {L('https://youtu.be/YnoXQUwImjU', '3 Rockers')}.
            </li>
            <li>Computer-vision-based visual inspection on the production line.</li>
            <li>Computer-vision-based human safety systems.</li>
          </ul>
        </div>

        <div className="entry">
          <p>
            <strong>Project Assistant</strong>, Indian Institute of Science (IISc), Bengaluru.
            <span className="year">Aug 2023 – Feb 2024</span>
          </p>
          <p>
            Tennis biomechanics — shot classification, contact detection, joint angles.
            Won 1st place at INCOFYRA (1 of 198 papers).
            Under Prof. SN Omkar (IISc), Abhilash Manu (EY).
          </p>
        </div>
      </section>

      {/* Projects */}
      <section id="projects">
        <h2>Projects</h2>

        <div className="project">
          <p className="project-desc">
            <strong>AI Interview Agent</strong> — takes interviews end-to-end with context of resume and JD.
            I made it say umm, hmm, ahh.
            {' '}{L('https://www.instagram.com/p/DIOyU2IBkTJ/', 'V1 demo')}, {L('https://www.instagram.com/p/DJQbRadTFSt/', 'V2 demo')} (V2 hit 5M+ views on Instagram).
          </p>
          <div className="project-media media-portrait">
            <video
              src={process.env.PUBLIC_URL + '/images/interview-agent-v2-demo.mp4'}
              controls
              playsInline
              preload="metadata"
            />
          </div>
        </div>

        <div className="project">
          <p className="project-desc">
            <strong>Voice Cloning (OpenVoice)</strong> — upload/record your voice, choose style, generate.
            {' '}{L('https://huggingface.co/spaces/naveenk-ai/openvoice_voicecloning_win', 'Try it')}.
          </p>
          <div className="project-media">
            <a href={process.env.PUBLIC_URL + '/images/openvoice-voice-cloning.png'} target="_blank" rel="noopener noreferrer">
              <img
                src={process.env.PUBLIC_URL + '/images/openvoice-voice-cloning.png'}
                alt="OpenVoice voice-cloning app — text to speak, style selector, reference audio, voice-similarity slider, and generated audio."
              />
            </a>
          </div>
        </div>

        <div className="project">
          <p className="project-desc">
            <strong>AgentML</strong> — fully local AI agent for end-to-end dataset analysis. EDA, feature engineering,
            model training, and multi-model debate. Uses Jupyter kernel for code execution, SQLite for sessions,
            open-source models via Ollama. No API keys, no cloud — runs entirely on your machine.
            {' '}{L('https://github.com/naveenkai/agentML', 'GitHub')}.
          </p>
          <div className="project-media">
            <div className="media-row">
              <a href={process.env.PUBLIC_URL + '/images/agentml_img1.png'} target="_blank" rel="noopener noreferrer">
                <img
                  src={process.env.PUBLIC_URL + '/images/agentml_img1.png'}
                  alt="AgentML chat analyzing the Titanic dataset with model reasoning and generated pandas code."
                />
              </a>
              <a href={process.env.PUBLIC_URL + '/images/agentml_img2.png'} target="_blank" rel="noopener noreferrer">
                <img
                  src={process.env.PUBLIC_URL + '/images/agentml_img2.png'}
                  alt="AgentML guided pipeline showing prep, EDA, features, training, and evaluation stages with live code output."
                />
              </a>
            </div>
            <a href={process.env.PUBLIC_URL + '/images/agentml_architecture.png'} target="_blank" rel="noopener noreferrer">
              <img
                src={process.env.PUBLIC_URL + '/images/agentml_architecture.png'}
                alt="AgentML system architecture: React/Zustand frontend over WebSocket and REST to a FastAPI backend, with a LangGraph agent pipeline, an isolated Jupyter kernel, and SQLite."
              />
            </a>
          </div>
        </div>
      </section>

      {/* Research */}
      <section id="research">
        <h2>Research</h2>

        <div className="entry">
          <p>
            <strong>Computer Vision & Biomechanics for Tennis Performance Analysis</strong>
          </p>
          <p>
            Shot classification (RNN), contact detection, joint angles.
            1st place at INCOFYRA — 1 of 198 papers. Prof. SN Omkar (IISc), Abhilash Manu (EY).
          </p>
          <div className="videos">
            <div className="video-card">
              <div className="video-wrap">
                <iframe src="https://www.youtube.com/embed/ovDj4v0vfWo?modestbranding=1&rel=0&controls=1" title="Shot classification" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
              </div>
              <span>Shot classification</span>
            </div>
            <div className="video-card">
              <div className="video-wrap">
                <iframe src="https://www.youtube.com/embed/RR2gIitjEw4?modestbranding=1&rel=0&controls=1" title="Phase classification" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
              </div>
              <span>Phase classification</span>
            </div>
            <div className="video-card">
              <div className="video-wrap">
                <iframe src="https://www.youtube.com/embed/an30U-l10D0?modestbranding=1&rel=0&controls=1" title="Contact detection" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
              </div>
              <span>Contact detection</span>
            </div>
          </div>
        </div>

        <div className="entry">
          <p>
            <strong>Gait Analysis with Multi 2D Cameras (POSTURA)</strong>
          </p>
          <p>
            2D clinical gait system. 85–95% keypoint accuracy, 98% gait phase detection.
            YOLOv8, ByteTrack, MediaPipe. 3 Rockers, perspective correction.
          </p>
          <div className="videos">
            <div className="video-card">
              <div className="video-wrap">
                <iframe src="https://www.youtube.com/embed/_PCpW9ioZr4?modestbranding=1&rel=0&controls=1" title="Keypoint extraction" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
              </div>
              <span>Keypoint extraction</span>
            </div>
            <div className="video-card">
              <div className="video-wrap">
                <iframe src="https://www.youtube.com/embed/YnoXQUwImjU?modestbranding=1&rel=0&controls=1" title="3 Rockers classifier" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
              </div>
              <span>3 Rockers classifier</span>
            </div>
            <div className="video-card">
              <div className="video-wrap">
                <iframe src="https://www.youtube.com/embed/tFITDzDOzkw?modestbranding=1&rel=0&controls=1" title="Masked output" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
              </div>
              <span>Masked output</span>
            </div>
          </div>
        </div>
      </section>

      {/* Writing */}
      <section id="writing">
        <h2>Writing</h2>
        {posts.slice(0, 3).map((p) => (
          <div className="entry" key={p.slug}>
            <p>
              <Link to={`/writing/${p.slug}`}><strong>{p.title}</strong></Link>
              {p.date && <span className="year">{formatDate(p.date)}</span>}
            </p>
            {p.blurb && <p>{p.blurb}</p>}
          </div>
        ))}
        {posts.length > 3 && (
          <p><Link to="/writing">All writing →</Link></p>
        )}
      </section>

      {/* Highlights */}
      <section>
        <h2>Misc</h2>
        <ul>
          <li>
            Interviewed 13 founders and researchers.
            {' '}{L('https://www.youtube.com/watch?v=2u8S3zcrhxA', 'My conversation with Robert Scoble')}.
          </li>
        </ul>
      </section>

      <footer>
        <p>
          {L('https://github.com/naveenkai', 'GitHub')}
          {' | '}
          {L('https://x.com/naveenk_ai', 'X')}
          {' | '}
          {L('https://www.linkedin.com/in/naveenkai/', 'LinkedIn')}
          {' | '}
          {L('https://www.youtube.com/@naveenk_ai', 'YouTube')}
        </p>
      </footer>
      </main>
    </div>
  );
}

export default Portfolio;
