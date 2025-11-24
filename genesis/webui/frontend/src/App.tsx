import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { GenesisProvider } from './context/GenesisContext';
import GenesisLayout from './components/GenesisLayout';
import CommandMenu from './components/CommandMenu';

export default function App() {
  return (
    <GenesisProvider>
      <CommandMenu />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<GenesisLayout />} />
        </Routes>
      </BrowserRouter>
    </GenesisProvider>
  );
}
