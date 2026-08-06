import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Portfolio from './Portfolio';
import Writing from './Writing';
import Post from './Post';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Portfolio />} />
        <Route path="/writings" element={<Writing />} />
        <Route path="/writings/:slug" element={<Post />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
