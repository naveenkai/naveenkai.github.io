import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Portfolio from './Portfolio';
import Writing from './Writing';
import Post from './Post';
import './App.css';

// Lazy so the markdown + KaTeX bundle only loads on /physics-of-ai pages
const PhysicsOfAI = lazy(() => import('./PhysicsOfAI'));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<Portfolio />} />
          <Route path="/writings" element={<Writing />} />
          <Route path="/writings/:slug" element={<Post />} />
          <Route path="/physics-of-ai" element={<PhysicsOfAI />} />
          <Route path="/physics-of-ai/:slug" element={<PhysicsOfAI />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
