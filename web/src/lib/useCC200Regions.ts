'use client';

import { useState, useEffect } from 'react';
import { loadCC200Regions, BrainRegion } from './cc200-regions';

export function useCC200Regions() {
  const [regions, setRegions] = useState<BrainRegion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function loadRegions() {
      try {
                // #endregion
        
        const loadedRegions = await loadCC200Regions();
        
                // #endregion
        
        if (mounted) {
          setRegions(loadedRegions);
          setLoading(false);
        }
      } catch (err) {
                // #endregion
        
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to load CC200 regions');
          setLoading(false);
        }
      }
    }

    loadRegions();

    return () => {
      mounted = false;
    };
  }, []);

  return { regions, loading, error };
}
