#import "template.typ": *

#show: academic-notes.with(
  // --- Required
  title: "GPU Computing",
  subtitle: "Unimi - Master's Degree in Computer Science",
  authors: (
    ("Luca Corradini", "LucaCorra02"),
    ("Matteo Zagheno", "zgn"),
  ),
  lang: "it",

  // --- Optional, uncomment to change
  repo-url: "https://github.com/LucaCorra02/GPU-Computing",
  course-url: "https://myariel.unimi.it/course/view.php?id=2942",
  year: "2025/26",
  lecturer: "Giuliano Grossi",
  // date: datetime.today(),
  // license: "CC-BY-4.0",
  // license-url: "https://creativecommons.org/licenses/by/4.0/",
  // heading-numbering: "1.1.",
  // equation-numbering: none,
  // page-numbering: "1",

  // --- Optional with language-based defaults, uncomment to change
  // introduction: auto,
  // last-modified-label: auto,
  // outline-title: auto,
  // part-label: auto,
  // note-title: auto,
  // warning-title: auto,
  // informally-title: auto,
  // example-title: auto,
  // proof-title: auto,
  // theorem-title: auto,
  // theorem-label: auto,
  // equation-supplement: auto,
  // figure-supplement: auto,
)

#part("Cuda Numba")
#include "chapters/Lezione1-Introduzione.typ"
#include "chapters/Lezione2-Gerarchia-Thread.typ"
#include "chapters/Lezione3-Warp.typ"
#include "chapters/Lezione4.typ"
#include "chapters/Lezione6.typ"
#part("PyTorch")
#include "chapters/Lezione-7.typ"
#include "chapters/Lezione10.typ"
