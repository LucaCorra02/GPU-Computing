#import "../template.typ": *

= Lezione 6

Lettura sequenziale permette di sfurttare la località della cache.

== Trasferimento di memoria

GPU esegue una copia locked della memoria del sistema. Passiamo dalla gestione di una memoria paginabile (dato non stabile nel tempo) possiamo gestire sulal GPU della memoria non paginabile (bloccata).

Ci sono delle API che permettono di allocare su HOST memoria non paginabile, una volta allocata possiamo fare trasferimenti asincroni.

Vorremo togliere lo stage intermedio, fissiamo esplicitamente la memoria, aumantando la velocità del DMA.

``` memcopyasync``` è la primitiva

=== Concurency overlap

Uno dei vantaggi è una concurincy overlap. Costruiamo degli stream indipendenti, ci possono essere trasferimenti su uno stream e esecuzioni in parallelo. IL DMA è unico, non posso fare trasferimenti paralleli ma posso far lavorare contemporaneamente CPU e DMA.

Problema, se allocassimo troppa pined memory potremmo andare a togliere un sacco di risorse. Se un singolo processo usa tante risorse, influisce sul tutto il sistema.

=== Gestione pined memory
//AGGIUNGRE MATRICI CON PIN MEMORY

== Stream

Uno stream è un unità di concorrenza in CUDA. Generare e gestire una coda FIFO di operazioni. Sono operazioni in sequenza dentro uno stream. Il vantaggio è che li stream sono indipendenti tra di loro (trasferimento e calcolo assieme) dipende dalla capacità della GPU.

Per usare uno stream, basta passarlo al kernell. Una volta creato uso uno stream con il suo identificatore.

Si sposano bene con la pinned memory. Nel momento in cui faccio il trasferimento posso decire lo stream su cui va.

Esistono due stream:
- Default o nulla  $"ID" = 0$
- Stream non nullo $"ID" > 0$
- Stream bullo, blocca tutti gli altri li stream, non può andare in concomitanza.

#nota()[
  Il default stream di default si sincronizza con gli altri stream.
]

Può aumentare l'occupancy. Nel momento in cui abbiamo taglie piccole di griglie, possono occupare solo alcuni SM, in vece di senqualizzare le cose l'idea è che arrivano diversi processi(griglie) e vengono gestiti parallelamente su stream diversi. Le applicazioni lavorano in maniera concorrente, se stiamo all'interno della stessa applicazione abbiamo il parallelismo.

``` cudaDeviceSynchronize``` si ripecuote a livello di sistema per tutti gli stream. é la sincronizzazione più _forte_ che abbiamo.
//Mettere immagine old vs new


#attenzione()[
  La creazione di uno stream deve avvenire prima del suo utilizzo. Di solito gli stream vengono creati e distrutti  con un loop
]

é possibile anche sincronizzare il singolo stream con l'host e non solo gli stream tra di loro.

QUERY = non bloccante, ma ci permette di indagare lo stato di uno stream, per capire ad esempio quante operazioni ha eseguito.

//Aggiugnere somam di vettori con stream

=== cuda pinned e stream

La memoria pined veiene allocata con ``` cuda.pinned.array(shape,dtype)```.
```py
arr = np.arrange(1_000_000, dtype=np.float32)
with cuda.pinned(arr):
  d_arr = cuda.to.device(arr)
  #kernel
  out = cuda.device.to.host(d_out)


```
L'array a viene pinnato e usato nel contesto come pinned.

Quando non usare pined:
- No uso di memoria GPU
- Non ci sono vantaggi oggettivi
- Troppi passaggi frequenti tra pinned e unpinned.

// capire cos'è
``` cuda.managed_array(shape,dtype)``` da usare con attenzione.

Uno stream si crea come ``` stream=cuda.stream()``` è indipendente dallo stream di default (non nullo). Per passarlo ad un kernel basta passarlo come parametro al kernel.

//aggiungere esempio pipeline loop in cuda
```py

  for i in range(num_streams):
    strat = i * chunk_size
    end = start + cunk_size

    #H2D copy
    cuda.to_device(h_in[start:end], to=d_buffers[i], stream=stream[i])

    scale_kernel[blocks, threads, stream[i]](d_buffers[i],210)
```
Se ho lanciato un loop sulle operazioni di stream, devo lanciare un syncronize in modo tale che tutto il lavoro su GPU sia finito metto un sincronize rispetto all'host.


