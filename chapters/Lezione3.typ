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

=== Block scheduling






=== Warp scheduling

== Branch control






Lo scheduler va a schedulare i warp per l'esecuzione fisica. 

Passaggi: 
- i blocchi vengono schedulati sull SM
- i warp di cui è costituito un blocco vengono schedulati e succissivamente eseguiti (servono 32 core all'interno di un SM per eseguire un warp), la perfezione è che avviene tutti in modo sincrono. Anche le memorie sono improntante a dimensione 32. Questa è la situazione che da il massimo througtput generale. 

Ci sono ragioni in cui un WARP può essere in wating in quanto tutte le sue 32 linee non sono pronte. 

//riguardare
Limitazione dello scheduling:
- Meno warp presenti nella lista dei pronti
- più latenza nell'accedere ai dati
- se saturo il numero di risorse per un thread singolo (tante varibili saturo i registri) lo sheduling genrale soffre. 

=== scheduling dei blocchi

la griglia è una collezione di blocchi (non c'è distinzione, noi la facciamo a livello logico), ogni blocco viene associato ad un SM. 

Gli SM consumano memoria condivisa. Il numero di blocchi può essere arbitrario (anche se limitato nel numero di thread), inoltre ogni blocco è indipendente nello scheduling. Ogni blocco lavora in locale in base alle risorse dell'SM. 

C'è un meccanismo di cooperazione tra blocchi si estende la sincronizzazione tra blocchi. 

=== Scheduling dei warp

ci possono essere più warp attivi all'interno del device. i warp vengono schedulati e eseguiti per istruzione per istruzione. 

i warp possono sincronizzarsi all'accesso in memoria attraverso delle primitive. 

Obbiettivo :
- ridurre la latenza avendo il numero massimo di warp attivi contemporaneamente.
$
  "Active warp" / "max possible warp per SM"
$

I branch rendono inefficiente il sistema. Un brench avviene nel codice, un thread ID segue una strda e un altro thread ID un'altra. é possibile riorganizzare i dati a livello di warp e non di thread. 

#attenzione()[
  Non è detto che c'è un assocazione 1:1 thread e dati, ovvero il primo dato corrisponde al thread con ID 1. Ma posso lavorare modulo la dimensione di warp 
]

=== Cluster

Possiamo associare più bloccho ad un cluster. I blocchi nello stesso cluster possono cooperare tra di loro. Aggungiamo un overhead di gestione per permette di gestire strutture dati più grandi. 

Nella situazione ci sono dei gruppi fisici di SM (GPC) che permettono questa cooperazione

=== Divergenza ed esecuzione

//Aggiugnere immagine ed esempio
i branch nel codice  spezzano i thread del warp in due. I thread non sono più eseguiti in modo parallelo ma in modo sequenziale. Di 32 linee paralle divido in due plotoni da 16

i branch vanno a inficiare sui warp, posso organizzare il codice per far si che non accada a livello di warp. 

//Aggiungere esempio
#esempio()[
  Spalma in due gruppi i warp. i branch *non hanno un impatto banale*. 

  Situazione ideale se avessi un array di 64 dovrei dire che i primi 32 si occupano dei pari e gli altri dei dispari. Posso sempre ragionare a gruppi di 32.  
]

=== Sincronizzazione

//aggiungere immagine
#nota()[
  Sincronizzazione livello di blocco principalmente (anche se esite per il warp )
]

Nel modello SIMT ogni thread può fare strade diverse. Ogni thread ha un flusso e può richiedere tempi diversi. Richiede sincronizzazione, quando si riparte con la prossima istruzione dobbiamo essere sicuri che i thread siano tutti allo stesso punto. 

in C esistono una serie di primitive: 
- ``` syncthreads``` = a un certo punto del codice kernel compare questa istruzione (interpretata da runtime CUDA). intriduce una barriera per i thread del blocco 
- ``` syncwarp``` = introduce sincronizzazione a livello di warp. 

//aggiungere immagine
Ogni thread ha un proprio progam counter PC (ognuno ha un registro). Di conseguenza i thread possono essere gestiti con divergenza e ri-convergere anche a livello di warp.  (poco interessante per il prof?)















