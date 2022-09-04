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
func caculate(a *int, b *[]int, c []*int, d *[][]int) {
	*a = 1
	*b = []int{1, 2, 3}
	for _, val := range c {
		fmt.Printf("%v ", *val)
	}
	fmt.Printf("\nsend caculate \n")
	*c[0] = 1

}
func main() {
	// chopsticks := make([]chan int, 5)
	// for i := range chopsticks {
	// 	chopsticks[i] = make(chan int, 1)
	// 	chopsticks[i] <- 1
	// }
	// for i := 0; i < 5; i++ {
	// 	go task(i, chopsticks, i)
	// }
	// time.Sleep(10000000000)
	// num := 4
	// a, b, c := 3, []int{1}, []*int{&num}
	// caculate(&a, &b, c)
	// fmt.Printf("a:%v\n", a)
	// fmt.Printf("b:%v\n", b)
	// for _, val := range c {
	// 	fmt.Printf("%v ", *val)
	// }
}
