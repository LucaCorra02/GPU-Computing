#import "../template.typ": *

= PyTorch on GPU

``` python
tensor_gpu = tensor_cpu.to(torch.device('cuda:0'))
tensor_gpu = tensor_cpu.cuda(0) #versione corta
```

Si tratta di un trasferimento di dati esplicito da CPU a GPU, il device GPU viene specificato tramite una stringa. Il problema è che possono fare delle operazioni miste, in particolare se vogliamo risparmiare operazioni possiamo concatenarle:
```py
x = x.to(device='cuda',dtype=np.float32)
```
Tramite la variabile `torch.set_default_device('cuda')`, posso cambiare il device di default (CPU). Le opraziomi `rand` e `zeros` ora eseguono una cuda malloc.

Scrivere le factory function, empty, zeros, ecc
Ci sono alcune operazioni che lavorano in place, ad esempio ```py
x = torch.rand(100,100,device='cuda:0')
y = torch.rand(100,device='cuda:0')

x.add_(y)#avviene un broadcasting implicito gestito da cuda
z = torch.rand(100,device='cpu')
#x.add_(z) z e x sono su due device diversi
```
La gestione del garbage avviene automaticamente ed è gestita a runtime (non dal programmatore).

I dati del modello possono essere spostati sulla gpu
```py
model = nn.Sequential(
  nn.Linear(784,256),
  nn.ReLu
)
model = model.to()
```
Allo stesso modo i dati nel training vengono elaborati a batch, i batch anche essi devono essere caricati sulla GPU. I singoli batch vengono trasferiti su device sia input che target
```py
for input,target in dataLoader:
  inputs = inputs.to(device)
  targets = targets.to(device)
  output = model(inputs) #risultati su GPU, vanno ritrasferiti
```

#nota()[
  Tutto il codice è molto portatile, può essere scalato per più cpu, più thread delle cpu in modo trasparente
]

Solitamente la memoria riservata e poco di più di quella allocata.

//aggiungere summary

= Stream

Solitamente ci sono una serie di _corsie_ su cui smalitre il traffico overo gli stream. Si possono sovrappore delle richieste di trasferimento e calcolo vero e proprio.

Esiste anche qui il default stream (di default le cose lanciate su GPU vanno qua), inoltre il default stream è mutualmente esclusivo, blocca tutti gli altri stream (non bello).

Usiamo la classe `torch.cuda.Stream` per la costruzione configurazione di stream.
```python

stream = torch.cuda.Stream(
  device = True,
  priority = 0, #priorità a livello di stream nello scheuling
)
```
#esempio()[
  ```py
  x = torch.randn(1000,1000, device='cuda')
  y = torch.matmul(x,x)

  stream1 = torch.cuda.Stream('cuda')
  stream2 = torch.cuda.Stream('cuda')
  with torch.cuda.stream(stream1):#passo l'esecuzione a stream1
    #op stream1
  with torch.cuda.stream(stream2):
    #op stream2
  ```
  Le esecuzioni di stream1 e stream2 sono davvero concorrenti
]
Possiamo avere anche degli stream annidati (creazione di stream dentro il contesto di un altro stream). Lo stream interno vive solamente nel contesto del with interno, successivamente il contesto passa a quello esterno

Esiste un metodo `wait.stream` che permette di sincronizarci con gli altri stream. Oltre agli eventi

== Trasferimento dati efficiente

Esistono due meccanismi ;
- `pinned_memory` trasferire i dati in pinned_memory
- `non_blocking` in modo asincrono, l'host non attende su quel trasferimento

=== Pin memory

`tensor.to(device, non_blocking)`. La pin memory può essere applicata ad un tensore per trasferirlo nella pinned.
//aggiunge pinned memory immage

Il codice che viene esegutio si blocca e aspetta `pin_memory` è quindi *blocking* per l'host.
Possono essere creati anche dei tensori direttamente in pin memory già inizializzati come vogliamo

== Eventi

All'interno di un kernel può essere messo un marker, ovvero `torch.cuda.Event()` e posso sincronizzare gli stream su un certo evento. Posso creare un interdipendenza tra eventi

`torch.cuda.syncronize` = sincronizza su tutto, host e strami, solitamente non si usa

solitamente di una `strea.sync` o `event.sync`.

```py
x_cpu = torch.rand(pinend=True)
x_gpu = device.to(non_blocking=True) #trasferimento asincrono
```

//aggiungere esempio sync tra stream

//aggiungere esercizio

Esempio in cui nelle fase di training uso degli stream in modo tale che se il batch N sta venendo computato, trasferisco il batch n+1 in GPU

Ho un punto unico di uso dei dati model. Devo sgnaciare solamente l'esecuzione del trasferimento su un altro stream, ne bastano due.

In questo caso non possiamo elaborare un chunk di dati su uno stream in maniera indipendente. Qui non va bene in quanto il model deve consumare tutti i dati che passano

Intando che il model elabora un cnhunk paralelamente carichiamo in memoria GPU (asincrono)

