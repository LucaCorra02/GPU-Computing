#import "../template.typ": *

= Lezione-6

In pythorch l'unità base sono i tensori. N-Data -> N indicies
Sono ricchi rispetto a i dati numpy. Contiene dei dati e ogni dato ha un tipo:
- Data
- Type
- Device (CPU o GPU)
Inoltre ci sono una serie di proprietà che servono per il machine learning.

Vengono usati per modellare tutta la parte _nuemerica_ :
- input
- output
- parametri

Tendenzialmente dato un dato di qualsiasi natura dovremmo andare a conventrilo in un tensore. Il tensore è una sorta di "wrap".

Un numpyarray quando diventa un tensore condivide la memoria (sono collegati per risparmiare memoria, puntatore alla stessa locazione di memoria).

Dal punto di vista delle dimensioni un tensore si legge seguendo le parentesi ``` [[[1,2,3],[4,5,6]]]``` è un tensore tri-dimensionale, la shape è `[1,3,3]`

== Tenosre

Per inizializzare un tensore usiamo:
```py
  import torch
  data = [[1,2],[3,4]]
  x_data = torch.tensor(data)
  print(x_data)
```
#nota()[
  Il tipo viene dedotto automaticamente dall'interprete.
]

Inoltre è possibile costruire un tensore da un vettore numpy e viceversa
```py
  np_array=np.array(data)
  x_np = torh.from_numpy(np_array)
```
Il tipo di dato e la shape vengono ereditati dal vettore numpy.

Riassumendo un tensore ha:
- Shape
- Datatype
- Device

#attenzione()[
  Pythorch di default usa i `float32` di default mentre numpy usa i `float64`. Se ereditiamo da numpy ci potrebbe essere un errore di conversione, bisgona riconvertire corretamente i dati.
]

=== Operazioni

Per quanto riguardo il calcolo matriciale ad esempio avviene con l'operatore `@`. Mentre l'operazione element-wise tra due vettor si può usare `*` come una normale moltiplicazione.

//aggiugnere esempio
Prodotto matriciale tra due tensori di batch diversi:
```
  a = torch.randn(2,4,5,4)
  b = torch.randn(2,1,4,3)
```

Le prime due dimensioni prendono il nome di batch $(2,4)$ e $(2,1)$ il cuore di questi due batch sono delle matrici di dimensioni $(5,4)$ e $(4,3)$. Le dimensioni dei batch non sono compatibili, faciamo *broadcasting*. Mentre la terza e la quarta dimensione sono il risultato di un prodotto matriciale.
```
a = torch.randn(2,4,5,4) #[B1, B2, 5, 4]
b = torch.randn(2,1,4,3) #[B1, 1, 4, 3]

x = torch.matmul(a,b)#[]
```
//aggiungere immagine
In questo prodotto avviene la replicazione, il missmatch viene coperto dal boradcasting. Ovvero replico la struttura dati con struttura minore tante volte quanto serve per arrivare alla dimensione dell'altro operando.

il matmul fissa le dimensioni `[...,n,m]*[...,n,m]`. Il batch finale avrà una dimensione $2,4$ mentre sulle due dimensioni avviene il prodotto matriciale così come lo conosciamo. La shape finale di `c.shape== (2,4,5,3)`.

//aggiungere esempio broadcasting con somma

Una somam di due vettori 1*4 e 4*1 è una matrice 4*4.

#nota()[
  Un vettore 1*n e n*1 non sono la stessa cosa, c'è un operazione di trasposizione in mezzo.
]

Un modo coinciso per fare compoment-wise product è `torch.ensum`. Prende come parametro una stringa sche descrive cosa deve essere moltiplicato e le operazioni:
#esempio()[
  ```
    A = torch.randn(2,3)
    B = torch.rand(3,4)
    C = torch.einsum(ik,kj->ij) # Dove k è la dimensione comune, sommiamo lungo k
  ```
  //aggiungere mapping formula

  - Nel caso di prodotto matrice bvettor è `ij,j -> i`
  - Nel caso di prodotto element-wise è `i,j,i,j->i,j`
  - Nel caso di matrice a più dimensioni otteniamo: `nij,njk -> nik`. Vado a moltiplicare solamente una dimensione delle matrici, scelgo una dimensione e moltiplico lungo quel asse i vettori.
]

La mean e la std di un immagine. In questo caso sto facendo la STD sul 3 ovvero il numero di canali, la media per il rosso, blue e verde `x.mean(dim=(0,2,3))`

#attenzione()[
  La scrittura `tensor.add_(5)` serve per le operazioni in place. Molto importante perchè sto modificando il dato su cui lavoro ma non vado ad espandere il dato di nuovo
]

== Normalizzazione

La batch normalization può essere fatta attraverso :
$(B,C,H,W)$ dove:
- B = batch size
- C = numero di canali
- H,W = dimensioni spazioli

Nei modelli deep i canali sono molto importanti in quanto continuano a cambiare gli embedding in spazi vettoriali di dimensione B.

Per ogni canale C vado a normalizzare, calcolo media e varianza su tutte le rimanenti variabili. Nella formula $c$ rimane sempre fisso:
$
  mu_c = 1 / ("BHW") sum_(b,h,w) x_(b,c,h,w) \
  sigma^2_c = 1 / ("BHW") sum_(b,h,w) x_(b,c,h,w) - mu_c
$
Il valore vinale è che ho cambiato il valore del dato $x$ da $hat(x)$. $epsilon$ è un numero molto piccolo per rimanere sopra 0 (?):
$
  hat(x)_(b,c,h,w) = (x_(b,c,h,w)-mu_c)/(sqrt(sigma^2_c + epsilon))
$
Tutto questo lo fa il moduli `batchNorm2d`.

== Tensor Reshaping

I data sono memorizati in maniera lineare in memoria. `view` e `reshape` cambiano la dimensione dei dati senza copiarli.

Quando creiamod dei tensori creiamo dei blocchi contigui.

```py
  x.arrange(12) # x è un tentore da 0 a 12
  x.view(3,4) # stessi dati di x riorganizzati in 3*4
```
#nota()[
  `View` lavora solamente su dati contigui.
  ```py
    x = torch.arange(12).view(3,4)
    y = x.trnspose(0,1)
    z = y.view(-1) #inferisce lui le dimensioni
  ```
  La matrice viene trasposta ed è per quello che non trova più la contiguita.

  Il concetto di contiguità significa: "I numeri che sono vicini nella matrice (logica) sono vicini anche nella striscia di memoria RAM (fisica)? 
  
  - Matrice originale ($3 times 4$): La prima riga è `0, 1, 2, 3`. In memoria sono seduti uno accanto all'altro? Sì (Contiguo).
  - Matrice trasposta ($4 times 3$): La prima riga ora è composta dai numeri `0, 4, 8`. In memoria sono seduti vicini? No! Tra lo 0 e il 4 ci sono sedute altre persone (1, 2, 3) che appartengono ad altre righe della nuova matrice.
  
  Perché view(-1) si rompe? Il comando view(-1) è "stupido": dice "Dammi tutti i dati in fila, dall'inizio alla fine della memoria". Se lo fai sulla trasposta, view andrebbe a leggere la memoria fisica nell'ordine originale: 0, 1, 2, 3....Ma la tua matrice trasposta logicamente dovrebbe iniziare con 0, 4, 8....C'è un disaccordo tra l'ordine fisico e quello logico. PyTorch se ne accorge e ti dice: "Non posso darti una vista piatta (view) perché se leggo la memoria di fila ti do i numeri nell'ordine sbagliato rispetto alla tua trasposta".
]

`reshape` è molto più flessibile, può implicare una ricopiatura dei dati se necessario.






#esempio()[
  Dato un batch di immagini, ovvero un tensore con $(B,3,H,W)$ dove $B$ numero di immagini, $3$ canali e $H e W$ dimensione dell'immagine.
  Vogliamo computare la meida tra tutti i canali $R,G,B$. Dove:
  $
    x -> hat(x) = (x-mu)/sigma
  $
  Dove prima abbatto i picchi dividendo per la deviazioen tramite $sigma$ poi vado a traslare sottraendo $mu$ (centro) rispetto alla media.

  Alla fine verifico che la media sia circa zero e la deviazione sia crica 1.
]



