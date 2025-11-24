import { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import MainContent from './MainContent';
import { useGenesis } from '../context/GenesisContext';

export default function GenesisLayout() {
  const [activeTab, setActiveTab] = useState('Tree');
  const { loadDatabases } = useGenesis();

  // Load databases on mount
  useEffect(() => {
    loadDatabases();
  }, [loadDatabases]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <MainContent activeTab={activeTab} />
    </div>
  );
}
