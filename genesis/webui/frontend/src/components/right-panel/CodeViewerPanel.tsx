import { useEffect, useRef, useCallback } from 'react';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import rust from 'highlight.js/lib/languages/rust';
import cpp from 'highlight.js/lib/languages/cpp';
import javascript from 'highlight.js/lib/languages/javascript';
import 'highlight.js/styles/github.css';
import { useGenesis } from '../../context/GenesisContext';
import './CodeViewerPanel.css';

// Register languages
hljs.registerLanguage('python', python);
hljs.registerLanguage('rust', rust);
hljs.registerLanguage('cpp', cpp);
hljs.registerLanguage('javascript', javascript);

export default function CodeViewerPanel() {
  const { state } = useGenesis();
  const { selectedProgram } = state;
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current && selectedProgram?.code) {
      hljs.highlightElement(codeRef.current);
    }
  }, [selectedProgram?.code]);

  const handleCopy = useCallback(async () => {
    if (selectedProgram?.code) {
      try {
        await navigator.clipboard.writeText(selectedProgram.code);
        // Could add a toast notification here
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    }
  }, [selectedProgram?.code]);

  const handleDownload = useCallback(() => {
    if (selectedProgram?.code) {
      const language = selectedProgram.language || 'py';
      const extension =
        {
          python: 'py',
          rust: 'rs',
          cpp: 'cpp',
          javascript: 'js',
        }[language] || language;

      const filename = `${selectedProgram.metadata.patch_name || 'code'}_gen${selectedProgram.generation}.${extension}`;
      const blob = new Blob([selectedProgram.code], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }
  }, [selectedProgram]);

  if (!selectedProgram) {
    return (
      <div className="code-viewer-panel empty">
        <p>Select a node from the tree to view code.</p>
      </div>
    );
  }

  const language = selectedProgram.language || 'python';
  const lines = selectedProgram.code.split('\n');

  return (
    <div className="code-viewer-panel">
      <div className="code-controls">
        <button onClick={handleCopy} title="Copy code to clipboard">
          📋 Copy
        </button>
        <button onClick={handleDownload} title="Download code as file">
          💾 Download
        </button>
        <span className="language-badge">{language}</span>
      </div>

      <div className="code-container">
        <div className="line-numbers">
          {lines.map((_, i) => (
            <span key={i}>{i + 1}</span>
          ))}
        </div>
        <pre>
          <code ref={codeRef} className={`language-${language}`}>
            {selectedProgram.code}
          </code>
        </pre>
      </div>
    </div>
  );
}
