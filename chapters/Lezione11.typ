#import "../template.typ": *

= PyTorch on GPU

A differenza degli array `numpy` che possono vivere solamente nella memoria host, i `tensori` possono esssere allocati anche su device.

Di *default* il device su cui vengono allocati i tensori è la memoria host. Per cambiare il device possiamo:
```py
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  torch.set_default_device('cuda') # Lo setta globalmente
```
#nota()[
  La stringa `cuda` si riferisce al device di defualt, tipicamente `cuda:0`. Se ci sono più device sulla macchina possiamo selezionarli come `cuda:1,cuda:2, ecc..`
]

è possibile anche cambiare device solo in un determinato contesto tramite `torch.cuda.device(device_id)`. Tale metodo infatti fornisce uno scope temporaneo in cui viene utilizzato il device indicato:
```py
with torch.cuda.device(0):
  tensor_gpu1 = torch.randn(1000,1000) # On gpu
  result = some_computation(tensor_gpu1)

#Device reverts to previous setting
```

== Trasferimento CPU -> GPU

Il trasferimenti avviene attraverso il metodo `.to()`. Esso può cambiare:
- Device
- Dype
- Layout
- Memory Format

Il trasferimento dati avviene in maniera *esplicita* da CPU a GPU (e viceversa), il device GPU viene specificato tramite una stringa:

```py
tensor_cpu = torch.randn(1000, 1000)
print(tensor_cpu.device)

# Pattern 1: Explicit device object
tensor_gpu = tensor_cpu.to(torch.device('cuda:0'))
print(tensor_gpu.device)

# Pattern 2: String specification
tensor_gpu = tensor_cpu.to('cuda:0')
print(tensor_gpu.device)

# Pattern 3: Legacy .cuda() method (still supported)
tensor_gpu = tensor_cpu.cuda(0)
print(tensor_gpu.device)

```
Possiamo andare anche ad effettuare delle operazioni aggiuntive durante il trasferimento, concatenandole tra di loro:

```py
  x = x.to(device='cuda',dtype=torch.float16)
```

Le *factory functions* (`empty`, `zeros`, `ones`, `rand`, ecc.) supportano il parametro `device` per creare tensori direttamente su GPU:
```py
  x = torch.zeros(100, 100, dtype=torch.float32 ,device='cuda:0')
  y = torch.rand(100, 100, device='cuda:0')
```
#nota()[
  Le device function sono molto utili in quanto permettono di rimuovere trasferimenti CPU->GPU non necessari. Allocazione in un'unica operazione.
]

== Operazioni

Alcune operazioni PyTorch possono essere *in-place* (`add_`,`mul_`,`relu_`,ecc). Esse modificano direttamente il tensore in memoria senza crearne uno nuovo:
```py
  x = torch.randn(100, 100, device='cuda:0')
  y = torch.randn(100, device='cuda:0') # Broadcastable shape, same device

  x.add_(y) # Valid: same device, broadcastable

  z = torch.randn(100, device='cpu')
  # x.add_(z) # RuntimeError: expected device cuda:0 but got cpu
```
In questo caso il *broadcasting* (se possibile) viene automaticamente gestito da pytorch.

La gestione del garbage collector avviene in maniera automatica, gestita a runtime (*non* dal programmatore).

== Modello su GPU

Possiamo trasferire un modello deep da CPU a GPU nel seguente modo. Internamente.
- Tutti i parametri (pesi e bias), vengono trasferiti sulla memoria della GPU
- Tutti i buffer (`BatchNorm`) vengono trasferiti
- La memoria allocata sul device per il modello rimane fino alla prossima passata del garbage collector

```py
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
  )
  model = model.to(device)
```

#nota()[
  PyTorch *non* permette operazioni tra host e device. Se il modello è allocato sulla GPU mentre l'input è sulla CPU verrà generato un `RunTimeError`.

  La *best practice* è utilizzare una variabile device definita globalmente.
  ```py
    model.to(device)
    tensor.to(device)
  ```
]
Per questo motivo è sempre necessario trasferire i dati sullo stesso device del modello. I dati nel training vengono elaborati a batch, anche essi devono essere caricati sulla GPU:

```py
  for inputs, targets in dataLoader:
    inputs = inputs.to(device)
    outputs = outputs.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    #Risultati sulla GPU
```

== Memoria della GPU

Possiamo vedere lo stato della memoria tramite due direttive:
- `torch.cuda.memory_allocated()`: Mostra la memoria allocata, utilizzata dai tensori
- `torch.cuda.memory:reserved()`; Mostra la memoria riservata

Inoltre, tramite la funzione `torch.cuda.empty_cache()` è possibile rilasciare la cache che la GPU non sta utilizzando, per renderla di nuovo disponibile. Questa operazione *non* ha effetto sulla memoria che sta venendo utilizzata dai tensori.

== Stream

Le moderne architetture dispongono di code hardware indipendenti su cui possiamo spalmare il traffico da smaltire. In pytorch uno *stream* definisce a quale di queste code di esecuzione hardware verrà inviata la successiva operazione CUDA.

PyTorch assegna di *default* le operazioni a uno _stream di default_. Tutte le operazioni lanciate su GPU di default si incanalano qui. Il $mr("problema")$ principale è che questo strem è *mutualmente esclusivo* viene sincronizzato implicitamente con tutti gli altri, creando cosi delle barriere di sincronizzazione che *riducono la concorrenza*.

In particolare, se non si creano nuovi stream si rischoa di serealizzare delle operazioni che potrebbero avvenire in parallelo come:
- Trasferimenti Host->Device
- Il lancio di kernel
- Trasferimenti Device->Host

I casi d'uso principale includono:
- *Overlap Copia - Calcolo*: Separando le operazioni in stream diversi, il motore DMA (responsabile dei trasferimenti via PCIe o NVLink) e il motore di calcolo (che esegue i kernel sui multiprocessori) possono operare simultaneamente e in modo indipendente.

- *Triple-Buffering*: Un pattern molto efficiente consiste nel dedicare tre stream separati: uno per i trasferimenti di dati verso la GPU (H2D), uno per i kernel di calcolo, e un terzo per riportare i risultati sulla CPU (D2H). Questa separazione abilita pipeline in cui l'upload, il calcolo e il download avvengono contemporaneamente.

- *Stram Based Prefetch*: Durante l'*addestramento*, uno stream dedicato alla copia può avviare il trasferimento del batch $N+1$ (dalla CPU alla GPU), proprio mentre lo stream di calcolo sta ancora elaborando il batch $N$. Questa tecnica incrementa notevolmente il throughput complessivo.

=== Creazione degli Stream

In pytorch viene utilizzata la classe `torch.cuda.Stream` per la costruzione e configurazione di stream. La creazione di stream è limitata dai limiti hardware della GPU, solitamente $32-64$ kernel concorrenti.

```python
  stream = torch.cuda.Stream(
    device='cuda:0', # Target device
    priority=0, # Priority level (lower = higher priority)
  )
```

#esempio()[
  Nell'esempio è il calcolo di $a$ e di $b$ avvengono in modo parallelo:
  ```py
  x = torch.randn(1000,1000, device='cuda')
  y = torch.matmul(x,x)

  stream1 = torch.cuda.Stream('cuda')
  stream2 = torch.cuda.Stream('cuda')

  with torch.cuda.stream(stream1): # passo l'esecuzione a stream1
    # operazioni eseguite su stream1
    a = torch.matmul(x,x)

  with torch.cuda.stream(stream2):
    # operazioni eseguite su stream2
    b = torch.matmul(x,x)
  ```
  Le esecuzioni di stream1 e stream2 sono concorrenti.
]

Possiamo avere anche degli *stream annidati* (creazione di stream dentro il contesto di un altro stream). Lo stream interno vive solamente nel contesto del `with` interno, successivamente il contesto passa a quello esterno:
```py
outer_stream = torch.cuda.Stream('cuda')
with torch.cuda.stream(outer_stream):
  # Opereazioni eseguite sullo stream outer
  x = torch.randn(1000, 1000)
  inner_stream = torch.cuda.Stream('cuda')
  with torch.cuda.stream(inner_stream):
    # Operazioni eseguite sul inner_stream
    y = torch.matmul(x, x)

  # Controllo passato all'outer_stream
  z = x + y # Nota: y può non essere pronta, serve sincronizzazione !
```
#nota()[
  Nell'esempio sopra $y$ potrebbe non essere pronta, richiede sincronizzazione.
]

La *sincronizzazione* avviene tramite il metodo `wait_stream()` che permette di sincronizzarci con gli altri stream, oltre agli eventi. In particolare per creare una dipendenza tra due stream:
```py
  torch.cuda.current_stream().wait_stream(copy_stream)
```

== Trasferimento dati efficiente

Il trasferimento efficiente tra CPU e GPU è essenziale. Potrebbe diventare un bottleneck se non fatto in maniera adeguata. PyTorch fornisce due meccanismi:
- *Pinned Memory* `tensor.pin_memory()`
- *Trasferimenti asincroni* `tensor.to(device, non_blocking=True)`. L'host non attende in questo tipo di trasferimento

#attenzione()[
  Usare la pinned_memory non sempre più efficiente, deve essere utilizzata con cautela.
]

=== Pin memory

In python esistono due tipi di memorie:
- *Pinned Memory*: La GPU può accedere direttamente alla memoria del host. Si tratta di un'area della RAM della CPU che viene bloccata, in modo che il sistema operatico non possa spostarla su disco.

- *Pageable Memory*: CUDA deve prima creare una copia del dato nella memoria pinned, successivamente procedere con il trasferimento.

*`.to(device)`* = Effettua il classico trasferimento da CPU a GPU. Il tensore inzialmente risiede in RAM nella parte *paginata*. Tuttavia questo trasferimento presenta il problema dello $mr("stagin")$: La GPU non può leggere direttamente dalla memoria paginabile. CUDA, sottobanco, crea una copia temporanea del tensore nell'area pinned. Solo dopo il tensore può essere trasferito sulla GPU. Si tratta di un'*operazione bloccante* (La CPU attende che il trasferimento sia completo)


*`.pin_memory()`* presenta due caratteristiche principali:
- *Accesso diretto*: Un tensore che risiede in questa zona di RAM può essere copiato dal motore DMA, senza bisogno dello stagin
- *Bloccante*: Si tratta di un'operazione bloccante per la CPU.

#nota()[
  Solitamente l'uso della pinned memory viene utilizzato con il parametro*`non_blocking=True`*. PyTorch cede il trasferimento dei dati su uno stream CUDA e restituire immediatamente il controllo alla CPU.

  Richiede che i dati di partenza che si vuole trasferire siano già nella `pinned_memory`.
]

#align(center)[
  #import "@preview/cetz:0.3.2": canvas, draw
  #canvas(length: 1cm, {
    import draw: *

    // Virtual Memory box
    rect((0, 0), (6, 7), stroke: 2pt, name: "vm")
    content((3, 6.5), text(11pt, weight: "bold")[Virtual memory])

    // Disk area
    rect((0.5, 4.5), (5.5, 6), fill: rgb("#f4e5a0"), stroke: 1pt, name: "disk")
    content((3, 5.5), [Disk])

    // RAM area
    rect((0.5, 0.5), (5.5, 4.2), fill: rgb("#f4e5a0"), stroke: 1pt, name: "ram")
    content((3, 3.8), [RAM (pageable)])

    // Pinned memory area inside RAM
    rect((2.5, 1), (5, 3.2), fill: rgb("#b8e6b8"), stroke: 1.5pt, name: "pinned")
    content((3.75, 2.9), text(9pt)[pinned memory])

    // GPU box
    rect((8, 2), (11, 5), stroke: 2pt, name: "gpu")
    content((10.3, 4.5), text(11pt, weight: "bold")[GPU])

    // ========== PERCORSO BLU (senza pinned memory - percorso lento) ==========
    // Tensori blu su disco
    circle((1.2, 5.5), radius: 0.15, fill: rgb("#7eb3e6"), stroke: 1pt)
    circle((1.5, 5.5), radius: 0.15, fill: rgb("#7eb3e6"), stroke: 1pt)

    // Tensori blu in RAM pageable
    circle((1.2, 3), radius: 0.15, fill: rgb("#7eb3e6"), stroke: 1pt)
    circle((1.5, 3), radius: 0.15, fill: rgb("#7eb3e6"), stroke: 1pt)

    // Tensori blu in pinned memory (copia intermedia CUDA)
    circle((3.2, 2), radius: 0.15, fill: rgb("#7eb3e6"), stroke: 1pt)
    circle((3.5, 2), radius: 0.15, fill: rgb("#7eb3e6"), stroke: 1pt)

    // Tensori blu in GPU
    rect((9, 4.2), (9.6, 4.6), fill: rgb("#7eb3e6"), stroke: 1pt)

    // Freccia blu: Disco → RAM
    line((1.35, 4.5), (1.35, 3.3), stroke: (paint: rgb("#7eb3e6"), thickness: 1.5pt), mark: (end: "stealth"))
    content((0.4, 3.9), text(7pt, fill: rgb("#7eb3e6"), weight: "bold")[])

    // Freccia blu: RAM → Pinned (copia intermedia)
    line((1.8, 3), (3, 2.2), stroke: (paint: rgb("#7eb3e6"), thickness: 1.5pt), mark: (end: "stealth"))
    content((2.3, 2.4), text(7pt, fill: rgb("#7eb3e6"), weight: "bold")[])


    // ========== PERCORSO ARANCIONE (con pinned memory) ==========
    // Tensori arancioni su disco
    circle((4.2, 5.5), radius: 0.15, fill: rgb("#e87030"), stroke: 1pt)
    circle((4.5, 5.5), radius: 0.15, fill: rgb("#e87030"), stroke: 1pt)

    // Tensori arancioni in RAM pageable
    circle((4.2, 3.5), radius: 0.15, fill: rgb("#e87030"), stroke: 1pt)
    circle((4.5, 3.5), radius: 0.15, fill: rgb("#e87030"), stroke: 1pt)

    // Tensori arancioni in pinned memory
    circle((4.2, 2.2), radius: 0.15, fill: rgb("#e87030"), stroke: 1pt)
    circle((4.5, 2.2), radius: 0.15, fill: rgb("#e87030"), stroke: 1pt)

    // Tensori arancioni in GPU
    rect((9, 3.3), (9.6, 3.7), fill: rgb("#e87030"), stroke: 1pt)

    // Freccia arancione: Disco → RAM
    line((4.35, 4.5), (4.35, 3.8), stroke: (paint: rgb("#e87030"), thickness: 1.5pt), mark: (end: "stealth"))
    content((5.2, 4.2), text(7pt, fill: rgb("#e87030"), weight: "bold")[])

    // Freccia arancione: RAM → Pinned Memory
    line((4.35, 3.2), (4.35, 2.5), stroke: (paint: rgb("#e87030"), thickness: 1.5pt), mark: (end: "stealth"))
    content((5.2, 2.9), text(7pt, fill: rgb("#e87030"), weight: "bold")[])

    // Freccia arancione: Pinned → GPU (DMA)
    line((5, 2.2), (8, 3.5), stroke: (paint: rgb("#e87030"), thickness: 2pt), mark: (end: "stealth"))
    content((6.5, 2.5), text(7pt, fill: rgb("#e87030"), weight: "bold")[DMA])

    // Legend
    let legend_y = -1.5

    // Pageable memory
    rect((0, legend_y), (0.8, legend_y + 0.5), fill: rgb("#f4e5a0"), stroke: 1pt)
    content((2.2, legend_y + 0.25), anchor: "west", text(9pt)[Pageable memory])

    // Non-pageable memory
    rect((0, legend_y - 0.8), (0.8, legend_y - 0.3), fill: rgb("#b8e6b8"), stroke: 1pt)
    content((2.2, legend_y - 0.55), anchor: "west", text(9pt)[Non-pageable (page-locked)])

    // Percorsi
    content((0, legend_y - 1.4), anchor: "west", text(10pt, weight: "bold")[Percorsi di trasferimento:])

    line(
      (0.2, legend_y - 1.9),
      (0.8, legend_y - 1.9),
      stroke: (paint: rgb("#7eb3e6"), thickness: 1.5pt),
      mark: (end: "stealth"),
    )

    content((2.2, legend_y - 1.9), anchor: "west", text(9pt)[`pinned_memory()`])

    line(
      (0.2, legend_y - 2.5),
      (0.8, legend_y - 2.5),
      stroke: (paint: rgb("#e87030"), thickness: 2pt),
      mark: (end: "stealth"),
    )
    content((2.2, legend_y - 2.5), anchor: "west", text(9pt)[`.to_device()`])

    // Transfer annotations
    content((1.6, 5), text(8pt, fill: rgb("#7eb3e6"), weight: "bold")[`pinned_memory()`])
    content((4.2, 5), text(8pt, fill: rgb("#e87030"), weight: "bold")[`.to_device()`])
  })
]


== Eventi

All'interno di un kernel può essere messo un marker, ovvero `torch.cuda.Event()` e posso sincronizzare gli stream su un certo evento. Posso creare un'interdipendenza tra eventi.

`torch.cuda.synchronize()` sincronizza su tutto, host e stream, solitamente non si usa.

Solitamente si usa `stream.synchronize()` o `event.synchronize()`.

```py
x_cpu = torch.rand(100, 100, pin_memory=True)
x_gpu = x_cpu.to('cuda', non_blocking=True) # trasferimento asincrono
```

#esempio()[
  Sincronizzazione tra stream usando eventi:
  ```py
  stream1 = torch.cuda.Stream()
  stream2 = torch.cuda.Stream()
  event = torch.cuda.Event()

  with torch.cuda.stream(stream1):
    x = torch.randn(1000, 1000, device='cuda')
    event.record() # registra l'evento su stream1

  with torch.cuda.stream(stream2):
    stream2.wait_event(event) # attende che stream1 raggiunga l'evento
    y = x + 1 # usa il risultato di stream1
  ```
]

=== Esempio: Sovrapposizione di trasferimento e computazione

Esempio in cui nella fase di training uso degli stream in modo tale che se il batch N sta venendo computato, trasferisco il batch N+1 in GPU:

```py
stream_compute = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()

for i, (inputs, targets) in enumerate(dataloader):
  # Trasferisco il batch corrente su GPU in modo asincrono
  with torch.cuda.stream(stream_transfer):
    inputs_gpu = inputs.to('cuda', non_blocking=True)
    targets_gpu = targets.to('cuda', non_blocking=True)

  # Attendo che il trasferimento sia completato
  stream_compute.wait_stream(stream_transfer)

  # Eseguo il forward pass
  with torch.cuda.stream(stream_compute):
    output = model(inputs_gpu)
    loss = criterion(output, targets_gpu)
    loss.backward()
```

Ho un punto unico di uso dei dati (model). Devo sganciare solamente l'esecuzione del trasferimento su un altro stream, ne bastano due.

In questo caso non possiamo elaborare un chunk di dati su uno stream in maniera indipendente. Qui non va bene in quanto il model deve consumare tutti i dati che passano.

Intanto che il model elabora un chunk, parallelamente carichiamo in memoria GPU (asincrono) il successivo.

