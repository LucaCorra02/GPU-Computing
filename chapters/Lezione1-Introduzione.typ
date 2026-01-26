#import "../template.typ": *

= Introduzione

== Perchè HPC?
L'obbiettivo è svolgere più operazioni possibili nell'unita di tempo.
- EXA = numero di operazioni floating point nell'unità di tempo
CPU = host, GPU = device, ad oggi sono collegate assime per abbatere il tempo della trasmissione (collo di bottiglia negli HPC).

Potremmo dire che la legge di Moore è mantenuta in vita dalle GPU per via dell'alto grado di parallellismo che porta ancora a notevoli miglioramenti.

=== GP-GPU (General Purpose Computing on GPU):
Utilizzare la GPU non solo per task specializzati relativi alla grafica digitale ma come dispositivo per l'esecuzione di codice "generico".

Anche in questo modello di calcolo la CPU gestisce l'orchestrazione del sistema. La GPU è trattata come device controllato dalla CPU.

== Motivazioni per il calcolo parallelo

Consideriamo il calcolo tra due matrici: il costo è $O(n^3)$ in un sistema di calcolo sequenziale. Siccome il risultato di ogni cella della matrice è indipendente, possiamo applicare un algoritmo parallelo. Le operazioni tra matrici, e altri calcoli simili, sono alla base di un qualsiasi modello di AI, da qui la necessità di renderli efficenti.

*CCN*: Reti che fanno lo stesso calcolo molte volte (tecnologia ormai superata), ad esempio operazioni che devono essere fatte per ogni pixel.

==== Parallelismo
Capacità di eseguire parti di un calcolo in modo concorrente (non cambia la semantica del calcolo, solo il modo in cui viene effettuato). Non tutti gli algoritmi sono parallelizzabili: alcuni totalmente, altri solo in parte, altri per nulla.

Con il parallelismo si possono:
- risolvere problemi più grandi nello stesso tempo di una computazione sequenziale di dimesioni più piccole
- risolvere un problema di dimensioni fisse in un tempo più breve

*Grado di parallelismo*: quanti sono grandi le unità perallele. Noi usermo i thread, ovvero unità di esecuzione costituita da una sequenza di istruzioni e gestiti dal sistema operativo (o da un sistema runtime), concorrono nell'accesso alle risorse.

=== Paradigma GP-GPU
- *Modello eterogeneo*: uso di CPU (host) e GPU (device) assieme per dare vita a un nuovo modello di calcolo
- *Separazione*: Parte sequenziale ed orchestrazione eseguita su CPU, parte parallela sulla GPU

La GPU necessità di trasferimenti diretti di memoria da parte della CPU. La CPU gestisce la sincronizzazione.

Il codice prodotto è costituito da due parti: bisogna esplicitare cosa va su CPU e cosa su GPU, ognuna sarà specializzata nell'esecuzione di determinati task.
I punti di sincronizzazione sono fondamentali per cordinare i due sistemi e spesso vanno inseriti esplicitamente nel codice.

Il calcolo parallelo può essere realizzato in vari modi:
- *Parallelismo dei dati*: suddivisione dei dati in parti uguali ed elaborazione simultanea di ogni parte su più processori (moltiplicazione matriciale)
- *Parallelismo di task*: suddivisione del lavoro in attività indipendenti e l'assegnazione simultanea di queste attività a diversi processori
- *Parallelismo di istruzioni*: suddivisione delle istruzioni di un programma in parti indipendenti e l'esecuzione simultanea di queste istruzioni su più processori

Non tutti i dati sono adatti al parallelismo, dipende dalla dimensione e dal tipo. Il modello di calcolo più adatto varia in base al tipo del problema.

=== Parallel computing

==== Modello PRAM
- Modello a memoria condivisa
- Processori indipendenti ma sincronizzati tra loro

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Processori
    let proc-y = 2
    let proc-spacing = 1.5
    for i in range(4) {
      let x = i * proc-spacing - 2.25
      rect((x - 0.3, proc-y - 0.4), (x + 0.3, proc-y + 0.4), fill: blue.lighten(60%), stroke: blue)
      content((x, proc-y), text(size: 9pt, $P_#i$))
    }

    // Memoria condivisa
    let mem-y = 0
    rect((-3, mem-y - 0.6), (3, mem-y + 0.6), fill: purple.lighten(70%), stroke: purple)
    content((0, mem-y), text(size: 10pt, [*Memoria Condivisa*]))

    // Connessioni
    for i in range(4) {
      let x = i * proc-spacing - 2.25
      line((x, proc-y - 0.4), (x, mem-y + 0.6), stroke: (paint: olive, dash: "dashed"))
    }

    // Etichette
    content((-3.5, proc-y), text(size: 9pt, [Processori:]))
  }),
  caption: [Modello PRAM: processori indipendenti con memoria condivisa],
)

==== Tassonomia di Flynn
- *SISD* (Single Instruction Single Data):
  - No parallelismo, un'istruzione alla volta
  - Modello della CPU tradizionale (single thread)
- *SIMD* (Single Istruction Multiple Data):
  - Computazione vettoriale
  - Unica istruzione su più dati. Ciascuna unità di calcolo lavora e accede a dati diversi
  - Istruzioni vettoriali
- *MISD* (Multiple Instruction Single Data):
  - Unità di computazione con accesso a memoria privata privata di programma e memoria globale condivisa
  - Ogni unità esegue istruzioni diverse sullo stesso dato
  - Nessuno sviluppo commerciale noto
- *MIMD* (Multiple Instruction Multiple Data):
  - Più unità di computazione con accesso indipendente a istruzioni e dati
  - Processori multicore, sistemi distribuiti

=== SIMT (Single Instruction Multiple Threads)
Oltre la tassonomia classica:

- Ogni thread può seguire dei flussi di istruzioni distinti dagli altri. Eseguono lo stesso codice ma nei punti di branch possono prendere strade diverse (devono essere applicati a dati distinti)

== Nvidia GPU

=== CUDA Core
Unità di base che che esegue solamente operazioni artimetiche.

=== SM (Streaming Multiprocessor)
Contiene un gruppo di CUDA core.

=== Warp
È l'unità fondamentale di esecuzione, consiste in 32 thread paralleli che in ogni istante eseguono tutti la stessa istruzione.

=== Tensor Core
È un'unità aritmetica analoga al CUDA core ma può fare più operazioni parallele su un gruppo di dati, ad esempio il prodotto matriciale tra matrici di dimensioni contenute (una matrice grande  può essere divisa in sotto-matrici).

Nvidia tesla T4 (la useremo in colab)

MIG architettura // TODO: sistemare qua

#figure(
  image("../resources/nvidia-gpu-arch.png", width: 80%),
  caption: [
    Typical NVIDIA GPU architecture.
    Hernández et al., 2013. Accelerating Fibre Orientation Estimation from Diffusion Weighted Magnetic Resonance Imaging Using GPUs. PloS one. 8. e61892. 10.1371/journal.pone.0061892.
    #link(
      "https://www.researchgate.net/figure/Typical-NVIDIA-GPU-architecture-The-GPU-is-comprised-of-a-set-of-Streaming_fig1_236666656",
    )[Source]
    #link("https://creativecommons.org/licenses/by/4.0/")[CC BY 4.0]
  ],
)

=== CUDA (Compute Unified Device Architecture) // TODO: sistemare da qui in poi (eventualmente con esempi di codice)
Framekork di Nvidia per il calcolo parallelo. Ecosistema compatibile con svariate librerie, tra cui _pyTorch_.

Compilando un codice CUDA (estensione _.cu_) usiamo il compilatore _NVCC_:
- Prende in ingresso il sorgente e produce del _fat-binary_ comprendente sia la parte host sia device.
- JIT: il codice viene prima compilato in PTX (Pseudo-Assembly CUDA) poi il driver lo compila in codice binario prima dell'esecuzione.

==== Sintassi CUDA

- Qualificatore _global_, l'esecuzione della funzione etichettata viene demandata alla GPU. _threadIdx_ è l'identificatore unico del thread.

- `<<<1,10>>>` esprime la configurazione del kernel, in questo caso la GPU lancia 10 thread che eseguono tutti lo stesso kernel.

- `cudaMalloc()` alloca memoria sulla GPU.

- `cudaMemcpy()` memcpy da host a device e viceversa.

Esempio di esecuzione:
- Copio i dati CPU $->$ device
- Eseguo somma su device
- Copio il risultato device $->$ CPU

== Numba CUDA

Operazioni su GPU tramite python (non tutto ma solo delle primitive).

Interoperabilità tra _NUMBA_ e _Numpy_.

Si usano i *decoratori*, ovvero funzioni che modificano altre funzioni, definiti dalla sintassi `@DecoratorName`

Numba, tramite un apposito decoratore, compila una funzione python per renderla eseguibile su GPU.

```py
  from numba import cuda
  @cuda.jit
  def kernel(a):
    pass
```

Se antepongo il decoratore `@cuda.jit` a una funziona, la funzione diventa un kernel.

Una volta definiti i kernel, li lanciamo con l'opportuno decoratore, specificando il numero di thread, il numero di blocchi e gli argomenti.

Utilizziamo `cuda.synchronyze()` per aspettare che l'esecuzione del programma lanciato sul device termini.

=== Somma parallela di vettori

Il default di python è 64 bit se definisco un intero. In GPU significa invocare delle specifiche unità di calcolo dei 64 che se non ci sono vengono simultate dai 32 (molto più lento). Bisogna specificare sempre il tipo.

I kernel come in C non restiutiscono valore.

Il kernel è sempre asincrono, l'istruzione successiva al kernel viene eseguita dalla CPU. I trasferimenti di memoria invece sono implicitamente sincroni.
