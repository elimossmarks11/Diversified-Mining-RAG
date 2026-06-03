## Extraction Method Comparison: Docling vs Spatial Clustering

Both extraction outputs for the same ANTAM financial highlights table 
were embedded with MiniLM and evaluated against 4 domain-specific queries.

| Query | Docling | Spatial Clustering | Winner |
|---|---|---|---|
| Gross profit 2013 | 0.355 | 0.230 | Docling |
| Non-controlling interests 2015 | 0.323 | 0.157 | Docling |
| Owners of parent 2016 | 0.219 | 0.101 | Docling |
| Total assets 2016 | 0.378 | 0.164 | Docling |

Docling won all 4 queries with a mean cosine similarity margin of 0.156. 
This confirms that extraction quality directly impacts retrieval quality — 
Docling's structured markdown preserves row/column relationships that 
MiniLM can encode meaningfully, while spatial clustering's flat text 
representation loses the semantic structure the embedding model relies on.

Running Docling on CPU takes longer than a spatial clustering approach. 
However, when run without OCR, it takes roughly 14 seconds per table 
page which is not unreasonable. For a production system with GPU 
access, Docling is clearly preferable. For CPU-only deployment, spatial 
clustering remains a viable strategy with known retrieval quality 
degradation on table-heavy pages.