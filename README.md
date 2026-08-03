# proyecto_finca_raiz_colombia

Bayesian analysis of the Bogotá residential property market, using listing data
collected from [FincaRaíz](https://www.fincaraiz.com.co).

Universidad Externado de Colombia — Estadística Bayesiana, cuarto semestre.

## Status

| Part | State |
|---|---|
| Data acquisition ([scraper/](scraper/)) | **Working.** Full venta run: 31,252 listings at 91.3% coverage |
| Cleaning / validation | **Working.** Runs inside the scraper pipeline (stages 4–6) |
| Exploratory analysis ([notebooks/](notebooks/)) | Not started |
| Bayesian models ([src/](src/)) | Not started |
| Shared helpers ([utils/](utils/)) | Not started |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere

make install                    # runtime dependencies
make install-dev                # ruff, for the PEP 8 gate

make scrape                     # arriendo listings  (~45 min)
make scrape-venta               # for-sale listings  (~70 min)
```

`make help` lists every target. Run commands from the repository root — the
scraper imports the project-wide settings object from [config.py](config.py).

## Layout

```
config.py           single Settings object (pydantic-settings, reads .env)
Makefile            quality gate + scraper entry points
ruff.toml           PEP 8 rules (line length 79)
requirements.txt    runtime dependencies
requirements/       dev.txt - lint tooling

scraper/            the data pipeline - see scraper/README.md
  pipeline.py       stage orchestration; one command runs 0-6
  stages/           s0_discovery ... s6_qa
  config.yaml       everything site-specific lives here
tests/scraper/      97 offline tests over saved fixtures
tools/              clean.py, mk_help.py - used by the Makefile

data/               scraped datasets (gitignored)
notebooks/          exploratory analysis
src/                models
utils/              shared helpers
outputs/plots/      figures
outputs/tables/     result tables
out-<op>-<city>/    pipeline working directories (gitignored)
```

## The dataset

The scraper emits a validated JSON artifact plus a flat CSV per query. Measured
on the completed `venta` / `bogota` run:

```
pages fetched   1,935          fetch errors        4
parsed          39,206         parse errors        0
duplicates      7,954          schema rejections   0
final           31,252         coverage           91.3%
```

Per listing: price, currency, administration fee, built and private area,
stratum, bedrooms, bathrooms, parking, floor, age bracket, construction state,
property type, latitude/longitude, locality, neighbourhood, address, and an
amenities list drawn from the site's 229-entry vocabulary. Full field mapping is
in [scraper/README.md](scraper/README.md).

Composition of that run: apartamento 12,337 · casa 9,135 · oficina 2,488 ·
bodega 2,185 · local 1,693 · apartaestudio 1,313.

### Three things to know before modelling

- **`age_years` is always null, deliberately.** The site publishes an age
  *bracket* ("más de 30 años"), never an exact year. Use `age_bracket_code` as
  an ordinal factor; a midpoint would be invented precision.
- **Nulls are informative, not missing at random.** The site writes `0` for
  values it does not publish, and the scraper restores those to null rather
  than treating them as real zeros. The rate depends on the operation, which is
  itself a signal:

  | Field | null in `arriendo` | null in `venta` |
  |---|---|---|
  | `admin_fee` | 75.6% | 43.5% |
  | `floor` | 55.9% | 50.5% |
  | `parking_spaces` | 63.5% | 34.5% |

  Treat these as *not disclosed* and consider modelling the disclosure
  mechanism rather than dropping or zero-filling the rows.
- **Coverage is measured, not assumed.** 91.3% on the venta run. The remainder
  is unreachable because the site's result ordering is unstable at depth — see
  the explanation in [scraper/README.md](scraper/README.md). Any market-level
  total from this data is a lower bound.

Datasets are gitignored (`*.csv`, `out-*/`), so a clone has no data until you
run `make scrape`.

## Code quality

```bash
make quality     # ruff check + format check across notebooks, scraper, src, utils
make test        # 97 offline tests
make validate    # both
make format      # auto-fix
```

`make quality` checks notebooks cell by cell. Line length is PEP 8's 79 —
raise it in [ruff.toml](ruff.toml) if it fights the analysis code.

## Before scraping at scale

Read [REVIEW_TOS.md](REVIEW_TOS.md). It covers the site's terms of use and
Ley 1581 de 2012 / Decreto 1377 de 2013. In short: the scraper never records
contact details, defaults to a conservative rate, and stops rather than
attempting to evade a block — but holding and publishing the data is still your
call to make deliberately.

## License

[MIT](LICENSE) © 2026 Juan Pablo Carrillo Hortua
