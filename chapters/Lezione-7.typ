#import "../template.typ": *

= PyTorch

Si tratta di una libreria che permette di andare a costruire modelli di deep learning. La maggior parte della libreria è scritta in `C++/Cuda`.

== Tensori

I tensori sono l'*unità base* di PyTorch. Si tratta di una generalizzazione di una scalare, vettore o matrice (una sorta di _wrap_). Aggiunge una serie di operazioni in più.\
Un tensore può avere diverse dimensioni:
- $0D -> "Scalari"$
- $1D -> "Vettori"$
- $2D -> "Matrici"$
- $3D+ -> "Tensori a grande dimensioni"$

In particolare un tensore, contiene:
- *Dati*, valori che incapsula
- *Type* (`dtype`), i dati che contiene avranno un certo tipico. Il tipo determina: precisione, utilizzo di memoria, operazioni valide
- *Device*, se risiede su `CPU` o `GPU`. Possono quindi essere eseguiti sulla GPU e possono sfruttare l'accelerazione hardware

Vengono usati per modellare tutta la parte _nuemerica_ del nostro modello: input, output e iperparametri.

Per inizializzare un tensore da dei dati, possiamo:
```py
  import torch
  data = [[1,2],[3,4]]
  x_data = torch.tensor(data)
  print(x_data)
```

#nota()[
  Il tipo viene dedotto automaticamente dall'interprete.
]

Inoltre è possibile costruire un tensore da un array `numpy` e viceversa:
```py
import numpy as np
data = [[1,2],[3,4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
```
Il vantaggio è che le prorpeità (`shape,datatype`) del vettore `numpy` vengono *ereditate* dal `tensor`, a meno che non vengano sovrascritte:
```py
x_ones = torch.ones_like(x_data) # [[1,1],[1,1]]
x_rand = torch.rand_like(x_data, dtype=torch.float)
#[[1.2,0.3],[0.23,2.3]]
```
#attenzione()[
  Quando si converte un vettore `numpy` ad un `tensor` ci sono due aspetti da considerare:
  - `PyTorch` di default usa i `float32`, mentre `numpy` usa i `float64`. Se ereditassimo direttamente da `numpy` ci potrebbe essere un errore di conversione, bisgona *riconvertire* corretamente i dati.

  - Un `numpy` array quando diventa un `tensor`, continua a *condividere la memoria* (puntatore alla stessa locazione di memoria). Se si effetuano dei cambiamenti sul `tensor`, influiscono anche sull'array `numpy`.
]

=== Shape

La dimensione di un `tensor` si leggono dall'esterno verso l'interno. Ad esempio:
```py
  tensor(
    [[
      [1,2,3],
      [4,5,6],
      [7,8,9]
    ]]
  )
  torch.size([1,3,3]) # dim 0,1,2
```
In questo caso è un tensore 3D, c'è una dimensione $3*3$.

== Operazioni

Di default le operazioni tra tensori possono essere eseguite sia sulla `CPU` che sulla `GPU`, generalmente i tensori vengono allocati inizialmente sulla `CPU`.

Esistono inoltre degli operatori caricati, ad esempio:
- `@`: utilizzato per *operazioni di algebra lineare*, come il prodotto matriciale
- `*`: utilizzato per *element-wise product*, ad esempio prodotto degli elementi cella per cella di un vettore

=== Broadcasting

Si tratta di un'operazione fatta in automatico da `PyTorch` quando i *batch* (prime due dimensioni) di due tensori non corrispondono:
```py
a = torch.randn(2, 4, 5, 4) # [B1, B2, 5, 4]
b = torch.randn(2, 1, 4, 3) # [B1, 1, 4, 3]
c = torch.matmul(a, b)
print("Shape of c:", c.shape) # [2, 4, 5, 3]
```
Nell'esempio la dimensione dei batch di $a$ e $b$ non corrispondono, in particolare:
- $a "batch"(2,4)$
- $b "batch"(2,1)$
#figure(
  image("../assets/broadcasting.png", width: 65%),
  caption: [
    Rappresentazione dei tensori $a$ e $b$ (input)
  ],
)
Siccome le dimensioni dei batch non corrispondono viene fatto *broadcasting* (in modo implicito). Il tensore con dimensione minore viene espanso tante volte quanto serve per arrivare alla dimensione dell'operando più grande, in modo da croprire così il missmatch.\
Nell'esempio sorpa il la seconda dimensione del batch del tenosore $b$ viene posta a $4$. Il tensore $c$ risultante avrà una `c.shape == (2, 4, 5, 3)`, dove:
- $2 ->$ batch esterno
- $4 ->$ gruppo interno del batch
- $5 ->$ numero di righe per ogni matrice
- $3 ->$ numero di colonne per ogni matrice

#figure(
  image("../assets/broadcasting-result.png", width: 65%),
  caption: [
    Rappresentazione dei tensori $a$ e $b$ (input)
  ],
)

#attenzione()[
  Non tutte le forme di tensori sono compatibili. Affinché il broadcasting funzioni, `PyTorch` confronta le dimensioni dei due tensori partendo da *destra verso sinistra* (dall'ultima dimensione alla prima). Due dimensioni sono *compatibili* se:
  - Sono uguali
  - Una delle due è $1$.

  Se un tensore ha meno dimensioni dell'altro, `PyTorch` aggiunge virtualmente delle dimensioni $1$ a sinistra.
]









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

== Numpy e Tensor

Attenzione condividono memoria. Può essere pericoloso, nel molemto in cui passo a un tensore da un numpy array e faccio operazione sul tensore allora modifico anche l'array numpy originale
//aggiugnere esempio

`unsqueexe(dim)` = aggiunge nuove dimensioni di dimensione 1 alla posizione dim.
`squeeze(dim)` = operazione contraria, rimuove degli $1$ dalle dimensioni

La concatenazione permette di concatenare tensori diversi su una certa dimensione.

`torch.stack`



