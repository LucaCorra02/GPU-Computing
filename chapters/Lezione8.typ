#import "../template.typ": *
= Lezione 8

I modelli sono in grado di catturare cose al di la dell'osservabile.

Funzione di attivazione: Sono funzioni che hanno due asintoti (da + infinito a - infitio). Per ogni valore x tendono ad essere in saturazione tra -1 e 1. La parte centrale è "meno convinta". Servono per arrivare a definire delle regioni non lineari (curvature dello spazio) in modo per distinguere gruppi di dati in base alla categoria di appartenenza. In uno stesso spazio di embedding devo creare delle superfici di separazione. Nella regione centrale il modello ha incertezza.
- Tangente iperbolica

//aggiungere immagine

RELU (molto usata): Approssimazione universale basata sulla RELU. Data una funzione $f in C([a,b],R)$. Una funzione può essere approssimata come una combinazioni lineari di RELU:
- Relu = $max(o, x)$. Ogni volta che la x è negativa prendo 0. $R(w x+b)$ dipende dal valore della retta $w x+ b$. Se applicco una trasformazione alla variabile indipendete la funzione reagisce in modi diversi
- Prendo tante ReLu e vado a coprire la mia funzione $f$ qualsiasi. Ottento una curva spezzata.
- Vado a combianre la funzione rossa e blue assime, voglio che la seconda sia più vicina possibile alla funzione che voglio approssimare (grigia).

In generale più funzioni prendo meglio approssimo, con un certo costo di complicazione.

Tutto quello che abbiamo bisogno sono le funzioni di attivzioni. Limitazioni:
- il modello può essere molto complesso
- più tempo di training e maggiore quantità di dati per avere una accuratezza accetabile

Il teorema possiamo generalizzarlo come per ogni funzione continua $g:[0,1]^D -> R$ è possibile approssimarla con un livello nascosto (percetrone): $W in R^(K times D) - b in R^k . omega in R^K$. Aimentanto $K$ ovvero le unità nascoste abbiamo più possibilità di imparare, con un costo maggiore.

Il vantaggio è che avere dei modelli profondi rispetto a modelli flat è che le reti neurali gerarchiche (diversi livelli di profondità) sono in grado di estrarre più informazoni dai dati. Vengono sfruttati dei bias induttivi (lo decidiamo noi, bias) per apprendere pattern complessi.

Di solito si vede che nei primi livelli capiscono dei pattern molto basilari (edge delle immagini ad esempio) ma non capiscono l'immagine nel suo complesso. Sono gli ultimi livelli che lo fanno.

Il learning è una stima di parametri (sempre più efficaci)usando questi bias che mettiamo noi (embedding e livelli intermidi) per favorie l'apprendimento di una rappresentazione (embedding) gerarchici.

== Pythorc nn framework

Usermeo il modulo `torch.nn` essenziale per costruire modelli deep.
Troviamo come moduli:
- layer: tutti i vari _livelli_. Esiston ricorrenti, lineari, convoluzionali ecc.
- Funzione di attivazione.
- Livelli di normalizzazione
- Funzione di perdità o loss.

Quando creaiamo un nuovo modello è costruire una sotto-classe di `nn.Module`. Un modulo o layer vengono costruti in Pythorc come:
- Costruttore: definisce i layer in cui vogliamo suddividere il nostro modello
- Implementa un metodo forward. Dice come far fluire i dati attraverso i modelli e come utilizzarli.

Esiste un medoto built in in python come $"__call__"$ sono all'interno della classe e non vengono chiamati esplicitamente (tipo il costruttore). Il metodo call è un metodo che viene chiamato per far si che sia un wrapper per il forward, fa delle cose prima e dopo e in mezzo chiama il metodo forward.
Quando chiamiamo
```py
  def modelNN()
    pass
  model=ModelNN()
```
In realtà costruiamo un modello da una certa classe (usa il metodo built in `__init__`). Quando mandiamo il modello in esecuzione, applichiamo il forward con parametro $x$ `y=model(x)`.\

#nota()[
  Se chiamassi il metodo come `y=model.forward(x)` mancano delle untià funzionali.
]

Una trasformazione lineare Solitamente viene scritta come una trasformazione affine:
$
  y = A overline(x)+overline(b)
$
Dovne A è una matrice. Moltiplico le righe di A per il vettore colonna (x) e poi le sommo.
Se applico la trasposizoone ad entrambi i membri ottengo:
$
  y & = (A overline(x))^t+overline(b)^t \
  y & = overline(x)^t A^t + overline(b)^t
$
Equivale a scrivere (usando una prorpeità e per essere più efficiente da un punto di vista della rappresentazione):
$
  y = overline(x)W^t+overline(b)
$
Tuttavia dobbiamo definire:
- Input (batch_size, in_features)
- Output(batch_size, out_features)
Le features differiscono in quanto stanno nelle dimensioni della matrice $W$ e del vettore bias $b$
```py
  layer = nn.Linear(in_features=5, out_features=3)
  x = torch.random(2,5) #batch_size =2
  y = layer(x)
  y.shape # torch.Size([2,3])
```
L'importante è che il tensore matchi sulle `in_features`. Il risultato dipende dal numero di `out_features`.

Uso tipico con layer:
```py
  nn.linear(784,128)
  nn.ReLu()
  nn.linear(128,10) #compatibili, output matcha con input
```

=== Inizializzazione dei pesi

All'inizio la matrice dei pesi (che dovrà essere imparata) è inizializzata casualmente. Esiste una branca che studia come inizializzare i parametri, da dove si parte dipende poi come il modello evolve.
`nn.init.xavier_normal(layer.weight)` permette di creare una matrice inizializzata secondo una certa distribuzione.


