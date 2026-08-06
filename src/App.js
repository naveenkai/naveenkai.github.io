import { HashRouter, Routes, Route } from 'react-router-dom';
import Portfolio from './Portfolio';
import Writing from './Writing';
import Post from './Post';
import './App.css';

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Portfolio />} />
        <Route path="/writings" element={<Writing />} />
        <Route path="/writings/:slug" element={<Post />} />
      </Routes>
    </HashRouter>
  );
}

export default App;
