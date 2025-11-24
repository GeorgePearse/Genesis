import { useState, useRef, useEffect, useCallback } from 'react';
import { Database } from 'lucide-react';
import { useGenesis } from '../context/GenesisContext';

// Import existing panel components
import TreeView from './left-panel/TreeView';
import ProgramsTable from './left-panel/ProgramsTable';
import MetricsView from './left-panel/MetricsView';
import EmbeddingsView from './left-panel/EmbeddingsView';
import ClustersView from './left-panel/ClustersView';
import IslandsView from './left-panel/IslandsView';
import ModelPosteriorsView from './left-panel/ModelPosteriorsView';
import BestPathView from './left-panel/BestPathView';
import MetaInfoPanel from './right-panel/MetaInfoPanel';
import ParetoFrontPanel from './right-panel/ParetoFrontPanel';
import ScratchpadPanel from './right-panel/ScratchpadPanel';
import NodeDetailsPanel from './right-panel/NodeDetailsPanel';
import CodeViewerPanel from './right-panel/CodeViewerPanel';
import DiffViewerPanel from './right-panel/DiffViewerPanel';
import EvaluationPanel from './right-panel/EvaluationPanel';
import LLMResultPanel from './right-panel/LLMResultPanel';

interface MainContentProps {
  activeTab: string;
}

// Tabs that show a split view with a right panel
const SPLIT_VIEW_TABS = ['Tree', 'Programs', 'Metrics', 'Embeddings', 'Clusters', 'Islands', 'LLM Posterior', 'Path → Best'];

// Right panel tab options
const RIGHT_TABS = [
  { id: 'code-viewer', label: 'Code' },
  { id: 'diff-viewer', label: 'Diff' },
  { id: 'node-details', label: 'Node' },
  { id: 'evaluation', label: 'Eval' },
  { id: 'llm-result', label: 'LLM' },
];

// Min/max constraints for the right panel
const MIN_RIGHT_PANEL_WIDTH = 300;
const MAX_RIGHT_PANEL_WIDTH = 1200;
const DEFAULT_RIGHT_PANEL_WIDTH = 500;

export default function MainContent({ activeTab }: MainContentProps) {
  const { state, setRightTab } = useGenesis();
  const { selectedRightTab, selectedProgram } = state;

  // Right panel width state with localStorage persistence
  const [rightPanelWidth, setRightPanelWidth] = useState(() => {
    const saved = localStorage.getItem('genesis-right-panel-width');
    return saved ? parseInt(saved, 10) : DEFAULT_RIGHT_PANEL_WIDTH;
  });

  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Save width to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('genesis-right-panel-width', rightPanelWidth.toString());
  }, [rightPanelWidth]);

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const newWidth = containerRect.right - e.clientX;

      // Clamp to min/max
      const clampedWidth = Math.min(
        Math.max(newWidth, MIN_RIGHT_PANEL_WIDTH),
        MAX_RIGHT_PANEL_WIDTH
      );

      setRightPanelWidth(clampedWidth);
    },
    [isDragging]
  );

  // Handle mouse up to stop dragging
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  // Add/remove global event listeners
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Start dragging
  const handleDividerMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  // Double-click to reset to default
  const handleDividerDoubleClick = () => {
    setRightPanelWidth(DEFAULT_RIGHT_PANEL_WIDTH);
  };

  const renderLeftPanel = () => {
    switch (activeTab) {
      case 'Tree':
        return <TreeView />;
      case 'Programs':
        return <ProgramsTable />;
      case 'Metrics':
        return <MetricsView />;
      case 'Embeddings':
        return <EmbeddingsView />;
      case 'Clusters':
        return <ClustersView />;
      case 'Islands':
        return <IslandsView />;
      case 'LLM Posterior':
        return <ModelPosteriorsView />;
      case 'Path → Best':
        return <BestPathView />;
      default:
        return <TreeView />;
    }
  };

  const renderRightPanel = () => {
    switch (selectedRightTab) {
      case 'code-viewer':
        return <CodeViewerPanel />;
      case 'diff-viewer':
        return <DiffViewerPanel />;
      case 'node-details':
        return <NodeDetailsPanel />;
      case 'evaluation':
        return <EvaluationPanel />;
      case 'llm-result':
        return <LLMResultPanel />;
      default:
        return <CodeViewerPanel />;
    }
  };

  const renderSinglePanel = () => {
    switch (activeTab) {
      case 'Meta':
        return <MetaInfoPanel />;
      case 'Pareto Front':
        return <ParetoFrontPanel />;
      case 'Scratchpad':
        return <ScratchpadPanel />;
      case 'Node':
        return <NodeDetailsPanel />;
      case 'Code':
        return <CodeViewerPanel />;
      case 'Diff':
        return <DiffViewerPanel />;
      case 'Evaluation':
        return <EvaluationPanel />;
      case 'LLM Result':
        return <LLMResultPanel />;
      default:
        return null;
    }
  };

  // If no data loaded, show empty state
  if (state.programs.length === 0) {
    return (
      <div className="flex-1 flex flex-col">
        <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-white">{activeTab}</h2>
            <div className="flex items-center gap-2">
              <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
                Export
              </button>
              <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
                Settings
              </button>
            </div>
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="mb-6 p-8 bg-gray-900 rounded-lg border border-gray-800">
              <Database className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-300 mb-2">
                No Database Selected
              </h3>
              <p className="text-sm text-gray-500 mb-6">
                Select a database to view evolution results and begin analysis.
              </p>
              <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium text-sm transition-colors">
                Load Database
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Check if this tab should show split view
  const isSplitView = SPLIT_VIEW_TABS.includes(activeTab);

  if (!isSplitView) {
    return (
      <div className="flex-1 flex flex-col">
        <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-white">{activeTab}</h2>
            <div className="flex items-center gap-2">
              <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
                Export
              </button>
              <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
                Settings
              </button>
            </div>
          </div>
        </div>
        <div className="flex-1 p-8 overflow-auto">{renderSinglePanel()}</div>
      </div>
    );
  }

  // Split view layout
  return (
    <div className="flex-1 flex flex-col">
      {/* Content Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium text-white">{activeTab}</h2>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
              Export
            </button>
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
              Settings
            </button>
          </div>
        </div>
      </div>

      {/* Split Content Area */}
      <div ref={containerRef} className="flex-1 flex overflow-hidden">
        {/* Left Panel - Main View */}
        <div className="flex-1 p-6 overflow-auto">
          {renderLeftPanel()}
        </div>

        {/* Resizable Divider */}
        <div
          onMouseDown={handleDividerMouseDown}
          onDoubleClick={handleDividerDoubleClick}
          className={`w-1 bg-gray-800 hover:bg-blue-500 cursor-col-resize transition-colors flex-shrink-0 group relative ${
            isDragging ? 'bg-blue-500' : ''
          }`}
          title="Drag to resize, double-click to reset"
        >
          {/* Visual indicator for drag handle */}
          <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-blue-500/20" />
        </div>

        {/* Right Panel - Code/Details View */}
        <div
          style={{ width: rightPanelWidth }}
          className="flex flex-col bg-gray-950 flex-shrink-0"
        >
          {/* Right Panel Header with Node Info */}
          {selectedProgram && (
            <div className="px-4 py-3 bg-gray-900 border-b border-gray-800">
              <div className="flex items-center gap-3 text-sm">
                <span className="font-medium text-white">
                  {selectedProgram.metadata.patch_name || `Program ${selectedProgram.id}`}
                </span>
                <span className="text-gray-500">Gen {selectedProgram.generation}</span>
                <span className="text-orange-400">
                  {selectedProgram.combined_score?.toFixed(4) ?? 'N/A'}
                </span>
                <span className={selectedProgram.correct ? 'text-green-400' : 'text-red-400'}>
                  {selectedProgram.correct ? 'Correct' : 'Incorrect'}
                </span>
              </div>
            </div>
          )}

          {/* Right Panel Tabs */}
          <div className="flex border-b border-gray-800 bg-gray-900">
            {RIGHT_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setRightTab(tab.id)}
                className={`px-4 py-2.5 text-sm font-medium transition-colors ${
                  selectedRightTab === tab.id
                    ? 'text-white border-b-2 border-blue-500 bg-gray-800'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Right Panel Content */}
          <div className="flex-1 overflow-auto p-4">
            {renderRightPanel()}
          </div>
        </div>
      </div>
    </div>
  );
}
