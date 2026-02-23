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

Solitamente la memoria riservata è poco di più di quella allocata. PyTorch gestisce automaticamente la memoria GPU tramite un allocatore interno.

= Stream

Solitamente ci sono una serie di _corsie_ su cui smaltire il traffico ovvero gli stream. Si possono sovrapporre delle richieste di trasferimento e calcolo vero e proprio.

Esiste anche qui il default stream (di default le cose lanciate su GPU vanno qua), inoltre il default stream è mutualmente esclusivo, blocca tutti gli altri stream (non bello).

Usiamo la classe `torch.cuda.Stream` per la costruzione e configurazione di stream.
```python
stream = torch.cuda.Stream(
  device='cuda:0',
  priority=0  # priorità a livello di stream nello scheduling (range: -1 a 0)
)
```
#esempio()[
  ```py
  x = torch.randn(1000,1000, device='cuda')
  y = torch.matmul(x,x)

  stream1 = torch.cuda.Stream()
  stream2 = torch.cuda.Stream()
  with torch.cuda.stream(stream1): # passo l'esecuzione a stream1
    # operazioni eseguite su stream1
    a = torch.randn(1000, 1000, device='cuda')
  with torch.cuda.stream(stream2):
    # operazioni eseguite su stream2
    b = torch.randn(1000, 1000, device='cuda')
  ```
  Le esecuzioni di stream1 e stream2 sono davvero concorrenti
]
Possiamo avere anche degli stream annidati (creazione di stream dentro il contesto di un altro stream). Lo stream interno vive solamente nel contesto del `with` interno, successivamente il contesto passa a quello esterno.

Esiste un metodo `wait_stream()` che permette di sincronizzarci con gli altri stream, oltre agli eventi.

== Trasferimento dati efficiente

Esistono due meccanismi:
- `pin_memory`: trasferire i dati in pinned memory
- `non_blocking`: trasferimento in modo asincrono, l'host non attende su quel trasferimento

=== Pin memory

`tensor.to(device, non_blocking=True)`. La pin memory può essere applicata ad un tensore per trasferirlo nella memoria pinned.

La pinned memory è una memoria RAM che non può essere spostata dalla memoria virtuale (page-locked). Questo permette trasferimenti DMA (Direct Memory Access) più veloci tra CPU e GPU, poiché il driver non deve prima copiare i dati in un buffer intermedio.

Il codice che viene eseguito si blocca e aspetta. `pin_memory` è quindi *blocking* per l'host.
Possono essere creati anche dei tensori direttamente in pin memory già inizializzati:
```py
x = torch.randn(100, 100, pin_memory=True)
```

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

