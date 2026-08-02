/* Spawn a command with ASLR disabled for that process image.
 *
 * The measurement harness pins benchmark processes' address layout so a
 * run's score is a property of the image, not of the launch draw
 * (mcts_mem interpreter/dispatch.md, 2026-08-01). Linux uses setarch -R,
 * whose personality children inherit; Darwin's spawn attribute applies
 * to one exec only, so this wrapper is applied per benchmark process via
 * the driver's --runner-prefix.
 *
 * POSIX_SPAWN_DISABLE_ASLR is not in the public header but has been
 * stable since 10.5; it is how debuggers pin inferior processes.
 */
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>

extern char **environ;

#ifndef POSIX_SPAWN_DISABLE_ASLR
#define POSIX_SPAWN_DISABLE_ASLR 0x0100
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: noaslr command [args...]\n");
        return 2;
    }
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_DISABLE_ASLR);
    pid_t pid;
    int rc = posix_spawnp(&pid, argv[1], NULL, &attr, &argv[1], environ);
    if (rc != 0) {
        fprintf(stderr, "noaslr: spawn failed: %d\n", rc);
        return 127;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 127;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 127;
}
