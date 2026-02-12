#import "../template.typ": *

= PyTorch autograd

== Cos'è un graidente

Partiamo da funzioni definite su uno spazio $D$ (input vettori) che mappano in un reale.

$gradient$ è un operatore che viene applicato a una funzioni $f$ (rappresnta una rete neurale):
$
  gradient f: R^D->R^D
$
#informalmente[
  un gradiente è un vettore di derivate parziali.
  Applichiamo la definizione di derivata solo che consideriamo il tasso di variazioni su una singola variabile, tenendo costanti le altri.
]
Si fa difierenziazione solo rispetto a una variabile ecc //aggiugnere formula

#esempio()[
  $
    f: R^2 -> R
  $
  Otteniamo un campo vettoriale:
  $
    delta f(x) = (alpha f) / (alpha x_1) hat(i)+ (alpha f) / (alpha x_2) hat(j)
  $
  Il grafico 3d ci dice che dato un vettore che parte in un punto ci dice il grado di pendenza di quella funzione. I vettori sono più o meno intesi (modulo del vettore). i vettori hanno direzioni pendenze e intesità diverse man mano che mi muovo nel dominio
]

Aggiungee latro esempio con due variabili.

=== Curve di livello

Sono importanti perchè quando usiamo il gradiente per raggiungere un punto di minimo in una funzione, usiamo le curve di livello. Un passo viene fatto da una cruva di livello ad un altra curva di livello (scegliendo una certa direzioni). Posso dstinguere le curve di livello in base a quelle che mi favoriscono la discesa verso il basso.

//aggiungere immagine campo vettoriale generato dal paraboloide

Il gradiente in ogni punto della funzione è perpendicolare alla tangente della curva di livello. Il gradiente viene interpretato come direzione

*Teo*: (direzione di massima discesa) //aggiungere
#informalmente()[
  Il segno serve altrimenti andre in salita
]
Lo sviluppo di taylor speiga il perchè:
$
  f(x) = f(x_0)+ gradient (f_x_0)^T dot (x-x_0)
$
#informalmente()[
  La funzione in un punto intorno da $x_0$. Se voglio usare la funzione di taylor per approssimare l'intorno di un punto devo ricorrere al gradiente moltiolicato (prodotto scalre)

  //riguardare
  Voglio valutare la funzione in x scritto come lo scostamento di x dal bho.\
  Vado a sostituire nella formula
  $
    x_i - alpha gradient(f_(x_o))
  $
]
Mi muovo verso un valore del dominio in cui la funzione ha valore minore. La migliore direzione è quella che coincide con $gradient f(X_0)$. Per il teorema se scelgo $d^*$ in questo modo ho una discesa ottimale in qualsiasi putno della funzione

=== algoritmo

//agiungere algoritmo
Il passo è $alpha_k$ (learning rate), potrebbe variari tutte le volte che ho un passo di iterazione tramite degli algoritmi

Criteri di stop. Vogliamo arrivare al minimo assoluto $gradient f(x_k) = 0$. Questa equazione è soddisfatta da un minimo relativo o minimi locali,oppure in caso di grafico a sella ho una variabile che mi fa salire e l'altra scendere.

Dobbiamo fissare una soglia $epsilon$. Se soo sotto la soglia mi fermo:

== Jacobiano
Quando abbiamo delle funzioni vettore->vettore sono delle funzioni non banali $R^n->R^m$ (layer dei modelli deep).

Lo jacobiano è la generalizzazione del gradiente per le matrici. Ogni componente della funzione $F(hat(x))$ è una funzione $f_x(overline(x)))$ a sua volta(composizoni di funzioni). Il gradiente viene esteso, deve differenziare rispetto alle funzioni e alle variabili:

La prima riga è la raccolta delle derivate parziali per la prima componente della funzione $F$. L'ultima riga è la derivata parziale rispetto all'ultima componente di $F$ grande.
#esempio()[
  Date $f_1 = x^2+y$ e $f_2 = x y$, ottengo $2x = (alpha f)/x$
]

#esempio()[
  Se prendo la funzione $h = g(overline(f))$. La f (composizoni di funzioni) è una variabile anche se deriva dal calcolo di $f_1(overline(x)), dots, f_k (overline(x))$.

  Se devo calcolare il gradiente di $h$ devo arrivare a moltiplicare lo jacobiao della funzione $f$ per la derivata priam di $g$:

  #esempio()[
    h è la composione di g e di f, posso dire che $h$ diepende da x, $h(overline(x))=g(f(x)))$.

    Posso usare la composizione in qualche modo //riguardare
  ]
]
=== regola della catena

Se predo due funzioni f e g e la loro composizione è derivabile ottendo $f'(g(x)) = r^k$

Posso fare la compsizione in quanto :$f(x_0)$ produce uan cosa di dim $m$ mentre $g$ prende $m$ e restituisce $k$:
$
  g(f(x_o)) -> z in R^k
$

#attenzione()[
  per calcolare i gradienti bisogna fare prodotti basati sugli jacobiani
]

== Loss

Una Loss :
$
  R^D -> R
$
è una funzione che mappa da $R^D$ a $R$ è l'ultimo step del procedimento visto fino adesso. L'obbiettivo è minimizzare questo funzione considerando il set di parameti $w$ come variabile.

Apllichiamo l'algoritmo di discesa dei gradienti per minimizzare la funzione

== Differenziazione automatica

Implementano il calcolo *esatto* (non approssimato) delle derivate di una funzione, si tratta di un meccanisomo di calcolo dei gradienti.

Vogliamo calcolare il gradiente della funzione loss avendo a disposizione un numero di elementi che mappano l'input sull'output (sono un numero elevato).

Potrei prendere ogni singolo elemento e farlo transistare nel mdoello -> calcolare la predizione -> caloclare l'errore -> modificare. Si tratta dell'approccio ideale. Tuttavia è meno robusto della discesa del gradiente stocastico e computazionalmente costoso

=== Graident stocastic

Scelgo a caso un elemento del dominio e sulal base dell'errore che si comette $L$, modifico i pesi. Un altro approccio più funzionante è basato su batch.

Lavora nel seguente modo:
$
  w_t+1 = w_t - n 1 / |B_t| sum_(n in B_t) gradient_w underbrace(l(f(x_n;w_t),y_n), "valore commesso medio")
$
Prendo un batch $B$ di tot elementi (piccola rispetto al dataset) e applico la variazione dei pesi sul valor medio di tutti i batch (non sul singolo esempalre)

Quello che succedo è che se prendo il gradiente medio calcolato sul batch è che non riesco ad avere la discesa ottimale in quel punto, tuttavia ho dei vantaggi :
- Molta velocità, meno computazioni
- Capacità di fare esapece dei minimi locali (in quanto è più rumoroso)

== Grafo computazionale

PyTorch implementa dietro le quinte autograd. Costuisce un grafo orientato (grafo diretto aicilico) che traccia tutte le operaizoni in tempo reale:
- I nodi sono le operaizoni ($+, -, *, "/"$)
- Gli archi sono l'interdipendenza tra questi nodi

Le aprime chiavi `requires_grad=True` associata ai tensori e l'operazione `backward()` ci permettono di fare il forward e modificare i pesi.

I gradienti vengono accumolati, ci sono un pool di gradienti che vengono utilizzati per modificare poi i pesi, vengono accumolati nel campo `.grad` di un tensore. La regola di Differenziazione viene applicata al grafo.

#esempio()[
  Regressione logistica (regressione per la classificazione).
  $D={x_i,y_i}$

  Se mi fermo al modello lineare ottengo un iperpiano, commette errori. Se prendo la funzione sigmoide o logistica ci permette di dire che se prendo una soglia (ad esempio $0.5$) ci permette di dire che se la rete emette $z$ (logits) da +/-inifnito la funzione di attivazione li riposta in un dominio più semplice, posso discriminarli in due classi 0 e 1.

  La funione di loos è la cross-entropy.

  Con $y=1$ (label True) mi trovo ad avere $-y log_2 sigma(z)$ se $y=0$ ho la funzinzone $1-log_2 sigma(z)$.

  Grado delle computazioni, prendo in input la variabile $x$ e $w$ (nel caso del forward la $x$ è variabile e $w$ costante, viceversa nella differenziazione). Esse danno forma alla variabile $z$ (ad esempio prodotto). La catenza forma un grafo ad alto livello. L'ultima parte è quella di decisione, il modello esce con dei valori

  //agiungere immagini
  A partre da $x$ e $w$ viene generata $u = x times w$. A questo punto lo sommo a $b$ (realizzo il livello lineare):
  $
    z = u + b
  $
  (Composizione di funzioni). In uscita avrò $hat(y) = theta(z)$ e infine calcolo la loss.

  Una volta ottenuto il valore di loos metto in atto la discesa del gradiente con la chain rule. La chain rule è data dalla derivata di tutte le variabili in gioco:

  - $t = (alpha l) / (alpha hat(y))$
  - $(alpha l) / a(alpha z)$
  - $(alpha z) / (alpha u)$
  - $(alpha u) / (alpha w)$

  la deriva di $(alpha L)/(alpha w)$ è data dal prodotto di tutte le derivate.
]

== SDG in PyTorch

`loss.backward()` fa tutto lui

`optimezer.step()` ci aggiunge una regola di aggiornamento.

```py
for epoch in range (n_epochs)
  for data, targat in dataloader:
    output = model(data)#costruisce un grafo
    loss = criteration(output, target)
    optimezer.zero_grad() #pulisce i buffer
    loss.backward()
    optimizer.step()#aggiorna i pesi
```

//aggoungere confronto mini-batch e singolo elemento
La disce del graideitne su un singolo elemnto alla volta è molto più smooth, tuttavia è molto costosa a livello computazionale.





