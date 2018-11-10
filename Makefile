all: first

first: first.c
		gcc  -std=c99 -Wall -Werror -fsanitize=address first.c -o first

clean:
	rm -rf first
