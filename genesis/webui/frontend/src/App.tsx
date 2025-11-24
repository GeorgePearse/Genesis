import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { GenesisProvider } from './context/GenesisContext';
import VisualizationLayout from './components/VisualizationLayout';
import './App.css';

export default function App() {
  return (
    <GenesisProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<VisualizationLayout />} />
        </Routes>
      </BrowserRouter>
    </GenesisProvider>
  );
}
