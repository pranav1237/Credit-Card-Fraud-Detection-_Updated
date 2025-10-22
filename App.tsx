import React, { useState, useMemo, useEffect } from 'react';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { DataTable } from './components/DataTable';
import { initialTransactions } from './data/transactions';
import type { Transaction } from './types';

export type Theme = 'dark' | 'light';

const App: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>(initialTransactions);
  const [theme, setTheme] = useState<Theme>('dark');

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  const originalData = useMemo(() => initialTransactions, []);

  const handleUpdateTransaction = (updatedTransaction: Transaction) => {
    setTransactions(prev => 
      prev.map(t => t.index === updatedTransaction.index ? updatedTransaction : t)
    );
  };

  return (
    <div className="min-h-screen font-sans">
      <Header theme={theme} toggleTheme={toggleTheme} />
      <main className="p-4 sm:p-6 lg:p-8">
        <div className="max-w-7xl mx-auto space-y-8">
          <Dashboard originalData={originalData} editedData={transactions} theme={theme} />
          <DataTable data={transactions} onUpdateRow={handleUpdateTransaction} />
        </div>
      </main>
    </div>
  );
};

export default App;