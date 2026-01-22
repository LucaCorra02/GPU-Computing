#import "../template.typ": *

// Da integrare con lezione 2
== Gerarchia della memoria

Panoramica logica: 
- ogni thread viene eseguito su un CUDA core ed ha uno spazio privato di memoria per registri, chiamate di funzioni. 

- a loro volta i thread sono collezionati   (a livello logico) in blocchi. Essi vengono eseguiti *concorrentemente* e possono cooperare attraverso barriere di sincronizzazione. Un blocco di thread usa una *shared memory* per la comunicazione intra-thread. 

- una grid è un array di thread block che *eseguono tutti lo stesso kernel*, legge e scrive su una memoria globale e sincronizza le chiamate di kernel tra di loro dipendenti.  

=== Mapping logico-fisico

Vale la seguente mappatura: 
- un thread viene mappato fisicamente su un *CUDA core* (unità esegutiva hardware)

- un blocco viene assegnato ad un singolo *SM* (non può essere spalmato su più SM). 
#nota()[
  Un singolo SM può eseguire più blocchi contemporaneamente (se possiede abbastanza risorse)
]

- una grid (insieme dei thread lanciati per un certo kernel) viene mappata sull'intero device. 

#figure(
  {
    import cetz.draw: *
    
    cetz.canvas({
      // Title labels
      content((0.9, 6), text(fill: black, weight: "bold", size: 11pt)[Software])
      content((5.6, 6), text(fill: black, weight: "bold", size: 11pt)[Hardware])
      
      // Vertical separator line
      line((4, 0), (4, 5.5), stroke: (paint: gray.lighten(50%), thickness: 1.5pt))
      
      // === First row: Thread -> CUDA Core ===
      
      // Thread (wavy line)
      bezier((0.6, 4.8), (0.9, 5.2), (0.75, 5), (0.7, 5.3), stroke: (paint: blue, thickness: 1.5pt))
      bezier((0.9, 5.2), (1.2, 4.8), (1.05, 5), (1.1, 4.7), stroke: (paint: blue, thickness: 1.5pt))
      content((0.9, 4.3), text(fill: black, size: 9pt)[Thread])
      
      // Arrow
      line((1.8, 4.8), (3.8, 4.8), mark: (end: "stealth"), stroke: (paint: blue.lighten(40%), thickness: 1.2pt))
      
      // CUDA Core (green rectangle)
      rect((5, 4.5), (6.2, 5.2), fill: rgb(50, 150, 50), stroke: black)
      content((5.6, 4.85), text(fill: black, size: 6.5pt)[CUDA Core])
      
      // === Second row: Thread Block -> SM ===
      
      // Thread Block (beige rectangle with wavy lines)
      rect((0.3, 2.8), (1.8, 3.9), fill: rgb(230, 200, 150), stroke: black)
      
      // Draw multiple wavy lines inside
      for i in range(7) {
        let x_offset = 0.45 + i * 0.18
        bezier((x_offset, 3.7), (x_offset + 0.07, 3.4), (x_offset + 0.035, 3.6), (x_offset + 0.035, 3.5), 
               stroke: (paint: blue, thickness: 0.8pt))
        bezier((x_offset + 0.07, 3.4), (x_offset + 0.14, 3.7), (x_offset + 0.105, 3.5), (x_offset + 0.105, 3.6), 
               stroke: (paint: blue, thickness: 0.8pt))
      }
      
      content((1.05, 2.4), text(fill: black, size: 9pt)[Thread Block])
      
      // Arrow
      line((2.2, 3.4), (3.8, 3.4), mark: (end: "stealth"), stroke: (paint: blue.lighten(40%), thickness: 1.2pt))
      
      // SM (multiprocessor with cores)
      rect((4.8, 2.7), (6.4, 4.1), fill: rgb(30, 30, 100), stroke: black)
      
      // Orange/red stripes at top and bottom
      rect((4.8, 3.95), (6.4, 4.1), fill: rgb(200, 100, 50), stroke: none)
      rect((4.8, 3.8), (6.4, 3.95), fill: rgb(255, 150, 50), stroke: none)
      rect((4.8, 2.7), (6.4, 2.85), fill: rgb(255, 150, 50), stroke: none)
      
      // Green cores grid
      for row in range(2) {
        for col in range(3) {
          rect((4.95 + col * 0.45, 2.95 + row * 0.4), (5.25 + col * 0.45, 3.2 + row * 0.4), 
               fill: rgb(50, 150, 50), stroke: black)
        }
      }
      
      content((6.9, 3.4), text(fill: black, size: 9pt)[SM])
      
      // === Third row: Grid -> Device ===
      
      // Grid (green background with multiple thread blocks)
      rect((0.2, 0.3), (3.2, 1.8), fill: rgb(180, 220, 150), stroke: black)
      
      // Three thread blocks inside
      for i in range(3) {
        let x_start = 0.35 + i * 0.95
        if i < 2 or i == 2 {
          rect((x_start, 0.45), (x_start + 0.7, 1.65), fill: rgb(230, 200, 150), stroke: black)
          
          // Wavy lines in each block
          for j in range(4) {
            let x_wave = x_start + 0.1 + j * 0.13
            bezier((x_wave, 1.5), (x_wave + 0.04, 1.2), (x_wave + 0.02, 1.4), (x_wave + 0.02, 1.3), 
                   stroke: (paint: blue, thickness: 0.8pt))
            bezier((x_wave + 0.04, 1.2), (x_wave + 0.08, 1.5), (x_wave + 0.06, 1.3), (x_wave + 0.06, 1.4), 
                   stroke: (paint: blue, thickness: 0.8pt))
          }
        }
      }
      
      // Dots between second and third block
      content((2.1, 1.05), text(fill: black, size: 12pt, weight: "bold")[...])
      
      content((1.7, 0), text(fill: black, size: 9pt)[Grid])
      
      // Arrow
      line((3.6, 1.05), (4.2, 1.05), mark: (end: "stealth"), stroke: (paint: blue.lighten(40%), thickness: 1.2pt))
      
      // Device (full GPU with multiple SMs)
      rect((4.5, 0.2), (7.8, 1.9), fill: rgb(150, 50, 50), stroke: black)
      
      // Two rows of SMs
      for row in range(2) {
        for col in range(3) {
          let x_sm = 4.65 + col * 0.95
          let y_sm = 0.35 + row * 0.8
          
          // SM block
          rect((x_sm, y_sm), (x_sm + 0.7, y_sm + 0.7), fill: rgb(30, 30, 100), stroke: black)
          
          // Top stripe
          rect((x_sm, y_sm + 0.58), (x_sm + 0.7, y_sm + 0.7), fill: rgb(200, 100, 50), stroke: none)
          rect((x_sm, y_sm + 0.5), (x_sm + 0.7, y_sm + 0.58), fill: rgb(255, 150, 50), stroke: none)
          
          // Bottom stripe
          rect((x_sm, y_sm), (x_sm + 0.7, y_sm + 0.08), fill: rgb(255, 150, 50), stroke: none)
          
          // Green cores
          for r in range(2) {
            for c in range(2) {
              rect((x_sm + 0.15 + c * 0.25, y_sm + 0.18 + r * 0.18), 
                   (x_sm + 0.3 + c * 0.25, y_sm + 0.3 + r * 0.18), 
                   fill: rgb(50, 150, 50), stroke: black)
            }
          }
        }
      }
      
      content((6.1, -0.15), text(fill: black, size: 9pt)[Device])
    })
  },
  caption: [
    Mappatura logico-fisica tra Software e Hardware.
  ]
)

== SIMT (Single instruction multiple thread)

Un *warp* è un gruppo di $32$ thread consecutivi (ID sequenziali) che vengono eseguti contemporaneamente.

Se i blocchi sono una suddivisione solamente a livello logico, un *warp* è un concetto fisico, ovvero come l'hardware organizza ed esegue fisicamente i thread. Ogni blocco viene linearizzato in più wrap seguendo un approccio *row-major*, prima sulla dimensione $x$, poi $y$ e infine $z$.

#informalmente()[
  L'hardware se ne sbatte dell'organizzazione a blocchi dei thread. Esso vede la griglia come sequenze di warp. L'ordine 2D "logico" viene linearizzato in $32$ thread sequenziali. 
]

L'ideale è che i blocchi siano multipli di $32$ in modo da permettere una buona mappatura sull'hardware. 

#esempio()[
  Supponendo un blocco di $128$ thread, verranno creati $4$ warp da $32$ thread.
]

#figure(
  {
    import cetz.draw: *
    
    cetz.canvas({
      // Thread Block 2D (4x4)
      
      
      content((2, 2.3), text(fill: black, size: 9pt, weight: "bold")[Thread Block (4×4)])
      
      // Define colors for each row (warp)
      let colors = (
        rgb(255, 230, 150),  // Row 0 - yellow
        rgb(255, 150, 150),  // Row 1 - red
        rgb(150, 150, 255),  // Row 2 - blue
        rgb(150, 220, 150),  // Row 3 - green
      )
      
      let cell_size = 0.7
      
      // Draw 4x4 grid of threads
      for row in range(4) {
        for col in range(4) {
          let x = 0.65 + col * cell_size
          let y = 4.65 - row * cell_size
          
          rect((x, y), (x + cell_size, y + cell_size),
               fill: colors.at(row),
               stroke: black)
          
          content((x + cell_size/2, y + cell_size/2),
                  text(fill: black, size: 7pt, weight: "bold")[T#sub[#row,#col]])
        }
      }
      
      // Linearized representation (row-major) - moved to right
      content((8.0, 4.3), text(fill: black, size: 9pt, weight: "bold")[Linearizzazione row-major])
      
      let linear_cell_w = 0.48
      let linear_start_x = 4.5
      let linear_y = 3.5
      
      // Draw linearized array with 16 threads
      for i in range(16) {
        let row = calc.quo(i, 4)
        let col = calc.rem(i, 4)
        let x = linear_start_x + i * linear_cell_w
        
        rect((x, linear_y), (x + linear_cell_w, linear_y + 0.5),
             fill: colors.at(row),
             stroke: black)
        
        content((x + linear_cell_w/2, linear_y + 0.25),
                text(fill: black, size: 6pt, weight: "bold")[T#sub[#row,#col]])
      }
      
      // Labels for warps
      content((linear_start_x + 2 * linear_cell_w, linear_y - 0.5), 
              text(fill: rgb(200, 180, 0), size: 8pt)[Warp 0])
      content((linear_start_x + 6 * linear_cell_w, linear_y - 0.5), 
              text(fill: rgb(200, 100, 50), size: 8pt)[Warp 1])
      content((linear_start_x + 10 * linear_cell_w, linear_y - 0.5), 
              text(fill: rgb(50, 50, 200), size: 8pt)[Warp 2])
      content((linear_start_x + 14 * linear_cell_w, linear_y - 0.5), 
              text(fill: rgb(50, 150, 50), size: 8pt)[Warp 3])
      
      // Row-major explanation
      content((7, 2.3), 
              text(fill: purple, size: 7pt, style: "italic")[Ordine: X → Y → Z])
    })
  },
  caption: [
    Linearizzazione row-major di un blocco $4*4$ in warp.\
    Ogni gruppo di $4$ thread consecutivi forma un warp (in questo esempio semplificato)
  ]
)

#nota()[
  La dimensione di un warp è una costante su varie architetture, in modo tale da garantire l'interoperabilità.
]

Idealmente, tutti i thread di un warp eseguono in parallelo la stessa istruzione (codice univoco dettato dal kernel) su dati diversi (modello SIMD). Tutti i thread *evolvono in parallelo su dati diversi*  

Il modello *SIMT*, introduce un PC (Program Counter) per ogni thread. In questo modo ognuno di essi può seguire un cammino di esecuzione diverso. Viene introdotto il problema della *wrap divergence*.

== Scheduling

La schedulazione avviene a $3$ *livelli di granularità* differenti: 
- schedulazione a livello di grid
- schedulazione a livello di blocchi
- schedulazione dei warp

=== Grid scheduling

Sulla GPU esiste un'unità hardware dedicata chiamata *GigaThread Engine*. Tale unità ha il compito di gestire la grid intera, andando a distribuire i blocchi sui vari SM disponibili.  

Entra in funzione quando l'Host (CPU) lancia un Kernel:
- se la GPU ha molti SM liberi, il GigaThread Engine assegna molti blocchi contemporaneamente.

- se la GPU è occupata o piccola, assegna pochi blocchi e mette gli altri in coda.

#nota()[
  Una volta che un blocco è assegnato a un SM, ci rimane fino alla fine della sua esecuzione. *Non* può "migrare" su un altro SM.
]

#figure(
  {
    import cetz.draw: *
    
    cetz.canvas({
      // Title at top
      content((1.7, 7.2), text(fill: black, weight: "bold", size: 10pt)[Grid of Thread Blocks])
      
      // Main grid of blocks (8x16) - made larger
      let block_size = 0.3
      let grid_start_x = 0.5
      let grid_start_y = 1.5
      
      // Define some highlighted blocks with their colors as array of tuples
      let highlighted = (
        ((2, 5), rgb(255, 100, 255)),   // Pink/Magenta
        ((3, 9), rgb(200, 150, 255)),   // Light purple
        ((5, 3), rgb(255, 150, 255)),   // Pink
        ((7, 11), rgb(150, 200, 255)),  // Light blue
        ((9, 2), rgb(100, 150, 255)),   // Blue
        ((11, 7), rgb(150, 100, 200)),  // Purple
      )
      
      // Draw main grid
      for row in range(16) {
        for col in range(8) {
          let x = grid_start_x + col * block_size
          let y = grid_start_y + (15 - row) * block_size
          
          let color = rgb(100, 180, 100)  // Default green
          
          // Check if this block is highlighted
          for item in highlighted {
            let coords = item.at(0)
            let h_color = item.at(1)
            if row == coords.at(0) and col == coords.at(1) {
              color = h_color
            }
          }
          
          rect((x, y), (x + block_size, y + block_size),
               fill: color,
               stroke: (paint: black, thickness: 0.5pt))
        }
      }
      
      // Dots below first grid
      content((2, 1), text(fill: black, size: 12pt, weight: "bold")[. . .])
      
      // Second smaller grid below (partially visible)
      let grid2_y = 0.2
      for row in range(2) {
        for col in range(8) {
          let x = grid_start_x + col * block_size
          let y = grid2_y - row * block_size
          
          rect((x, y), (x + block_size, y + block_size),
               fill: rgb(100, 180, 100),
               stroke: (paint: black, thickness: 0.5pt))
        }
      }
      
      // Right side - Active Thread Blocks label
      content((6.3, 7.2), text(fill: black, weight: "bold", size: 9pt)[Active Thread Blocks])
      
      // First SM box
      rect((5, 5.5), (7.5, 6.8), fill: rgb(200, 230, 200), stroke: (paint: black, thickness: 1pt))
      content((6.25, 6.15), text(fill: black, weight: "bold", size: 10pt)[SM])
      
      // Active blocks for first SM (3 small rectangles)
      let active1_colors = (rgb(255, 100, 255), rgb(200, 150, 255), rgb(255, 150, 255))
      for i in range(3) {
        rect((5.3, 6.4 - i * 0.28), (5.7, 6.65 - i * 0.28),
             fill: active1_colors.at(i),
             stroke: (paint: black, thickness: 0.5pt))
      }
      
      // Arrows from highlighted blocks to first SM
      line((grid_start_x + 5 * block_size, grid_start_y + 10.5 * block_size), 
           (5, 6.3), stroke: (paint: black, thickness: 0.8pt))
      line((grid_start_x + 9 * block_size, grid_start_y + 5.5 * block_size), 
           (5, 6.0), stroke: (paint: black, thickness: 0.8pt))
      line((grid_start_x + 3 * block_size, grid_start_y + 13 * block_size), 
           (5, 6.5), stroke: (paint: black, thickness: 0.8pt))
      
      // Dots between SMs
      content((6.25, 4.8), text(fill: black, size: 12pt, weight: "bold")[. . .])
      
      // Second SM box
      rect((5, 2.5), (7.5, 3.8), fill: rgb(200, 230, 200), stroke: (paint: black, thickness: 1pt))
      content((6.25, 3.15), text(fill: black, weight: "bold", size: 10pt)[SM])
      
      // Active blocks for second SM
      let active2_colors = (rgb(150, 200, 255), rgb(100, 150, 255), rgb(150, 100, 200))
      for i in range(3) {
        rect((5.3, 3.4 - i * 0.28), (5.7, 3.65 - i * 0.28),
             fill: active2_colors.at(i),
             stroke: (paint: black, thickness: 0.5pt))
      }
      
      // Arrows from highlighted blocks to second SM
      line((grid_start_x + 11 * block_size, grid_start_y + 3.5 * block_size), 
           (5, 3.3), stroke: (paint: black, thickness: 0.8pt))
      line((grid_start_x + 2 * block_size, grid_start_y + 10 * block_size), 
           (5, 3.0), stroke: (paint: black, thickness: 0.8pt))
      line((grid_start_x + 7 * block_size, grid_start_y + 8 * block_size), 
           (5, 2.8), stroke: (paint: black, thickness: 0.8pt))
    })
  },
  caption: [
    Schedulazione dei blocchi sui Streaming Multiprocessor (SM).\
    I blocchi colorati nella grid vengono assegnati dinamicamente agli SM disponibili
  ]
)


=== Block scheduling

I blocchi vengono assegnati in maniera *dinamica* agli SM (contiene molte unità funzionali e risorse) man mano che le risorse diventano disponibili. 

L'SM ha il compito di gestire uno o più blocchi. Un SM *accetta un nuovo blocco solo se ha abbastanza risorse* hardware (registri e Shared Memory) per ospitare tutti i thread di quel blocco.

#nota()[
  I blocchi in una griglia sono tra di loro *indipendenti*, non si hanno garanzie sull'ordine di esecuzione. 
]

Appena un blocco viene caricato sull'SM, viene logicamente suddiviso in Warps.

#attenzione()[
  é importante che un certo blocco *non* dipenda dal risultato di altri blocchi.

  La cooperazione inoltre avviene solamente all'interno del blocco (tranne casi particolari).
]

=== Warp scheduling

All'interno di ogni SM ci sono dei *Warp Scheduler*. Il loro compito è decidere quale warp deve eseguire un'istruzione in un certo istante (ciclo di clock).

Il meccanismo prende il nome di *latency hiding*:
- lo scheduler ha una lista di tutti i warp presenti sull'SM.

- controlla quali warp sono "pronti" (Ready) e quali sono "bloccati" (Stalled). Un warp è bloccato se, ad esempio, sta aspettando un dato dalla memoria globale. Se il warp $A$ sta aspettando la memoria, lo scheduler passa al warp $B$ che invece ha i dati pronti per fare calcoli.

- una volta scelto il Warp, i suoi 32 thread eseguono la *stessa istruzione* contemporaneamente sulle unità funzionali (core).


I warp possono sincronizzarsi nell'accesso alla memoria attraverso delle primitive. La situazione ideale è che avvenga tutto in modo sincrono. 

L' obbiettivo dello scheduler è *ridurre la latenza*, avendo il numero massimo di warp attivi contemporaneamente:
$
  "Active warp" / "max possible warp per SM"
$

#nota()[
  Se saturo il numero di risorse per un thread singolo (tante varibili che saturano i registri), lo sheduling genrale soffre.

  Un'alta occupazione non garantisce per forza delle buone performance.
]

Anche le memorie sono improntante a dimensione $32$.

== Branch control

I *branch* rendono inefficiente il sistema. Quando un branch viene eseguito esso causa una "divisione" del flusso di esecuzione. All'interno di un warp due thread potrebbe seguire percorsi diversi, rendendo necessaria la *sincronizzazione*. 

I gruppi di thread così creati, non sono più eseguiti in modo parallelo ma in modo sequenziale.


#esempio()[
  #figure(
  grid(
    columns: (0.5fr, 1.0fr),
    column-gutter: 1em,
    [
      // Left side - Code
      #set text(size: 11pt)
      ```c
      if(threadIdx.x % 2 == 0){
        a = r(t);
      }else{
        a = q(t);
      }
      y = f(a);
      ```
    ],
    [
      // Right side - Warp lanes visualization
      #import cetz.draw: *
      #cetz.canvas({
        let lane_width = 0.45
        let lane_height = 0.4
        
        // Title
        content((2, 4.8), text(fill: black, size: 10pt, weight: "bold")[Warp Lanes])
        
        // Draw lanes 0-5 and 31
        for i in range(6) {
          let x = i * lane_width
          let color = rgb(150, 200, 100)
          
          rect((x, 3.8), (x + lane_width, 4.2),
               fill: color,
               stroke: black)
          
          content((x + lane_width/2, 4),
                  text(fill: white, size: 8pt, weight: "bold")[#i])
        }
        
        // Dots
        content((6 * lane_width + 0.3, 4), text(fill: black, size: 11pt)[...])
        
        // Lane 31
        let x31 = 7.5 * lane_width
        rect((x31, 3.8), (x31 + lane_width, 4.2),
             fill: rgb(150, 200, 100),
             stroke: black)
        content((x31 + lane_width/2, 4),
                text(fill: white, size: 8pt, weight: "bold")[31])
        
        for i in range(6) {
          let x = i * lane_width
          let is_even = calc.rem(i, 2) == 0
          let color = if is_even { rgb(150, 200, 100) } else { gray.lighten(40%) }
          
          rect((x, 3), (x + lane_width, 3.4),
               fill: color,
               stroke: black)
          
          content((x + lane_width/2, 3.2),
                  text(fill: if is_even { white } else { gray }, size: 8pt, weight: "bold")[#i])
        }
        
        content((6 * lane_width + 0.3, 3.2), text(fill: black, size: 11pt)[...])
        
        // Lane 31 (odd - idle)
        rect((x31, 3), (x31 + lane_width, 3.4),
             fill: gray.lighten(40%),
             stroke: black)
        content((x31 + lane_width/2, 3.2),
                text(fill: gray, size: 8pt, weight: "bold")[31])
        
        
        
        for i in range(6) {
          let x = i * lane_width
          let is_odd = calc.rem(i, 2) == 1
          let color = if is_odd { rgb(150, 200, 100) } else { gray.lighten(40%) }
          
          rect((x, 1.6), (x + lane_width, 2),
               fill: color,
               stroke: black)
          
          content((x + lane_width/2, 1.8),
                  text(fill: if is_odd { white } else { gray }, size: 8pt, weight: "bold")[#i])
        }
        
        content((6 * lane_width + 0.3, 1.8), text(fill: black, size: 11pt)[...])
        
        // Lane 31 (odd - active)
        rect((x31, 1.6), (x31 + lane_width, 2),
             fill: rgb(150, 200, 100),
             stroke: black)
        content((x31 + lane_width/2, 1.8),
                text(fill: white, size: 8pt, weight: "bold")[31])
        
        
        for i in range(6) {
          let x = i * lane_width
          let color = rgb(150, 200, 100)
          
          rect((x, 0.8), (x + lane_width, 1.2),
               fill: color,
               stroke: black)
          
          content((x + lane_width/2, 1),
                  text(fill: white, size: 8pt, weight: "bold")[#i])
        }
        
        content((6 * lane_width + 0.3, 1), text(fill: black, size: 11pt)[...])
        
        // Lane 31 final
        rect((x31, 0.8), (x31 + lane_width, 1.2),
             fill: rgb(150, 200, 100),
             stroke: black)
        content((x31 + lane_width/2, 1),
                text(fill: white, size: 8pt, weight: "bold")[31])
        
      })
    ]
  ),
  caption: [
    Visualizzazione della *warp divergence*.\
    I $mg("thread")$ pari eseguono `r(t)` mentre i dispari sono mascherati.\
    Succesivamente i thread dispari eseguono `q(t)` mentre i pari sono mascherati.\
    Il *$mr("throughput viene dimezzato")$* perché servono $2$ cicli invece di $1$.
  ]
) <warp-lanes-divergence>
  


  
]




La soluzione é *riorganizzare i thread a livello di warp*, in modo che non si verifichino attese. 

#attenzione()[
  *Non* è obbligatorio che ci sia un assocazione 1:1 thread e dati, ovvero il primo dato corrisponde al thread con $"ID" 1$. L'idea è lavorare modulo la dimensione di warp.
]

L'$mg("obiettivo")$ è far sì che *tutti i thread dello stesso warp prendano la stessa decisione*:

- *Ordinamento dei Dati*: Ad esempio se stiamo processando un array dove alcuni elementi richiedono un calcolo pesante (ramo $A$) e altri no (ramo $B$), possiamo prima ordinare l'array raggruppando tutti gli elementi "pesanti" insieme e quelli "leggeri" insieme.

- *Mapping dei thread*: Se non è possibile spostare i dati, possiamo cambiare il modo in cui i thread scelgono su quale dato lavorare. L'idea è raggruppare thread che hanno un alta probabilità di seguire lo stesso percorso logico.

//TODO
=== Sincronizzazione

Nel modello *SIMT*, ogni thread può fare strade diverse. Ciascun threa può seguire un flusso "indipendente", richiedendo tempi diversi rendendo neccessaria . Richiede sincronizzazione, quando si riparte con la prossima istruzione dobbiamo essere sicuri che i thread siano tutti allo stesso punto. 

//aggiungere immagine
#nota()[
  Sincronizzazione livello di blocco principalmente (anche se esite per il warp )
]


in C esistono una serie di primitive: 
- ``` syncthreads``` = a un certo punto del codice kernel compare questa istruzione (interpretata da runtime CUDA). intriduce una barriera per i thread del blocco 
- ``` syncwarp``` = introduce sincronizzazione a livello di warp. 

//aggiungere immagine
Ogni thread ha un proprio progam counter PC (ognuno ha un registro). Di conseguenza i thread possono essere gestiti con divergenza e ri-convergere anche a livello di warp.  (poco interessante per il prof?)


=== Cluster

Possiamo associare più bloccho ad un cluster. I blocchi nello stesso cluster possono cooperare tra di loro. Aggungiamo un overhead di gestione per permette di gestire strutture dati più grandi. 

Nella situazione ci sono dei gruppi fisici di SM (GPC) che permettono questa cooperazione












