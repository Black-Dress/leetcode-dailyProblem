package main

import (
	"fmt"
	"time"
)

func task(id int, chopsticks []chan int, i int) {
	for {
		if i%2 == 0 {
			<-chopsticks[i]
			<-chopsticks[(i+1)%5]
		} else {
			<-chopsticks[(i+1)%5]
			<-chopsticks[i]
		}
		fmt.Printf("%d is eating\n", id)
		if i%2 == 0 {
			chopsticks[(i+1)%5] <- 1
			chopsticks[i] <- 1
		} else {
			chopsticks[i] <- 1
			chopsticks[(i+1)%5] <- 1
		}
		time.Sleep(1)
	}
}
func main() {
	chopsticks := make([]chan int, 5)
	for i := range chopsticks {
		chopsticks[i] = make(chan int, 1)
		chopsticks[i] <- 1
	}
	for i := 0; i < 5; i++ {
		go task(i, chopsticks, i)
	}
	time.Sleep(10000000000)
}
