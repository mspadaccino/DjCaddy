"""`core.guide`: il README ridotto alla guida che l'app mostra in Help.

Il rischio di filtrare invece di riscrivere è che il README si riorganizzi e
il filtro resti indietro in silenzio — via un capitolo di troppo, o dentro
uno che non doveva esserci. Qui si prova la meccanica su testi finti, e poi
si controlla il README VERO: che i titoli da togliere esistano ancora con
quel nome, e che quelli da tenere ci siano.
"""

import pytest

from core import guide

SAMPLE = """# Titolo

Apertura.

## Contents

- [Capitolo](#capitolo)

## Capitolo

Testo del capitolo, con un rimando a [i comandi](#command-line).

### Install

Da non mostrare.

### Sezione buona

Da mostrare.

## Command line

```bash
# questo cancelletto non e' un titolo
poetry run python cli.py
```

## Ultimo

Fine.
"""


@pytest.fixture
def filtered():
    return guide.guide(SAMPLE)


# --------------------------------------------------------------------------
# la meccanica
# --------------------------------------------------------------------------

def test_the_dropped_chapters_are_gone(filtered):
    assert "## Command line" not in filtered
    assert "## Contents" not in filtered
    assert "poetry run python cli.py" not in filtered


def test_the_dropped_section_is_gone_but_its_chapter_stays(filtered):
    assert "### Install" not in filtered
    assert "Da non mostrare" not in filtered
    assert "## Capitolo" in filtered
    assert "Testo del capitolo" in filtered


def test_a_dropped_section_does_not_swallow_the_next_one(filtered):
    """Il bug da cui nasce il test: una sezione tolta che si porta via
    anche quelle che la seguono, fino al capitolo dopo."""
    assert "### Sezione buona" in filtered
    assert "Da mostrare" in filtered


def test_the_chapter_after_a_dropped_one_comes_back(filtered):
    assert "## Ultimo" in filtered
    assert "Fine." in filtered


def test_a_hash_inside_a_code_block_is_not_a_heading():
    """Se lo fosse, il filtro leggerebbe i commenti di shell come titoli."""
    kept = guide.guide("## Capitolo\n\n```bash\n## Install\n```\n")
    assert "```bash" in kept and "## Install" in kept


def test_links_into_dropped_chapters_become_plain_text(filtered):
    """Un link che non porta più da nessuna parte è peggio di nessun link."""
    assert "(#command-line)" not in filtered
    assert "i comandi" in filtered


def test_links_that_still_land_somewhere_stay_links():
    kept = guide.guide("## Capitolo\n\nvedi [oltre](#ultimo)\n\n## Ultimo\n")
    assert "[oltre](#ultimo)" in kept


def test_the_anchor_rule_is_github_s():
    """Ogni spazio un trattino, non ogni GRUPPO di spazi: è la differenza
    fra `groove--read-this` e `groove-read-this`."""
    assert guide.anchor("Groove — read this one") == "groove--read-this-one"
    assert guide.anchor("BPM and key") == "bpm-and-key"


def test_contents_lists_chapters_and_sections_in_reading_order(filtered):
    assert guide.contents(filtered) == [
        (2, "Capitolo"), (3, "Sezione buona"), (2, "Ultimo")]


# --------------------------------------------------------------------------
# contro il README vero
# --------------------------------------------------------------------------

def test_every_title_the_filter_names_still_exists():
    """Se un titolo del README cambia, il filtro lo lascerebbe passare in
    silenzio: è esattamente ciò che questo test non permette."""
    titles = {title for _, title in guide._headings(guide.readme().read_text())}
    missing = [name for name in guide.DROPPED_CHAPTERS + guide.DROPPED_SECTIONS
               if name not in titles]
    assert not missing, f"titoli non più nel README: {missing}"


def test_the_real_guide_keeps_the_app_and_drops_the_code():
    kept = {title for _, title in guide.contents(guide.guide())}
    for name in ("Navigator", "Cue Finder", "Tag Maker", "File Analysis",
                 "Reference: what every number means"):
        assert name in kept
    for name in guide.DROPPED_CHAPTERS + guide.DROPPED_SECTIONS:
        assert name not in kept


def test_no_link_in_the_real_guide_points_nowhere():
    text = guide.guide()
    alive = {guide.anchor(title) for _, title in guide._headings(text)}
    dead = [a for a in guide._LINK.findall(text) if a[1] not in alive]
    assert not dead, f"link morti nella guida: {dead}"
