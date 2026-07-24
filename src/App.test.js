import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the portfolio at the index route', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: 'Naveen K', level: 1 })).toBeInTheDocument();
  expect(screen.getByText(/I build AI systems/i)).toBeInTheDocument();
});
