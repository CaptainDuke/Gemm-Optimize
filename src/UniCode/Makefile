# for Unicode


NEW := MyMM1

CFLAGS := -O2 -msse3 -mavx -mavx2
LDFLAGS := -lm


.c .o:
	@gcc -c $(CFLAGS) $<
.c .o:
	@gcc -c $(CFLAGS) $<

run: MyMM.o MyMMult.o
	gcc  $(CFLAGS) $(LDFLAGS) MyMM.o MyMMult.o -o run
	@rm MyMMult.o 
	@rm MyMM.o
	#@rm ./figs/output_$(NEW).m
	@echo "" > ./figs/output_$(NEW).m 
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	@./run >> ./figs/output_$(NEW).m 
	#@python plot.py 

clean:
	rm -f *.o
