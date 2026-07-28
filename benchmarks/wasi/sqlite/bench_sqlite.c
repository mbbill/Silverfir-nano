#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/bench.h"
#include "sqlite-amalgamation-3470200/sqlite3.h"

#define ROW_COUNT 1024
#define OPERATIONS_PER_UNIT 8

static sqlite3 *database;
static sqlite3_stmt *select_stmt;
static sqlite3_stmt *update_stmt;
static sqlite3_stmt *restore_stmt;
static int batch_error;
static volatile sqlite3_int64 result_sink;

static int execute(const char *sql) {
    char *message = NULL;
    int rc = sqlite3_exec(database, sql, NULL, NULL, &message);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite error: %s\n",
                message ? message : sqlite3_errmsg(database));
        sqlite3_free(message);
    }
    return rc;
}

static int step_for_id(sqlite3_stmt *statement, int id, int expect_row) {
    int rc;
    sqlite3_reset(statement);
    sqlite3_clear_bindings(statement);
    if (sqlite3_bind_int(statement, 1, id) != SQLITE_OK) return SQLITE_ERROR;
    rc = sqlite3_step(statement);
    if (expect_row) {
        if (rc != SQLITE_ROW) return rc;
        result_sink += sqlite3_column_int64(statement, 0);
        rc = sqlite3_step(statement);
    }
    return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/*
 * One unit operates on the same fixed database and row set. Every increment
 * is paired with a decrement, restoring the initial state before the next
 * unit while exercising prepared SELECT and UPDATE statements.
 */
static void sqlite_batch(long n, void *context) {
    long unit;
    (void)context;
    for (unit = 0; unit < n && !batch_error; unit++) {
        int operation;
        for (operation = 0; operation < OPERATIONS_PER_UNIT; operation++) {
            int id = 1 + operation * (ROW_COUNT / OPERATIONS_PER_UNIT);
            if (step_for_id(select_stmt, id, 1) != SQLITE_OK ||
                step_for_id(update_stmt, id, 0) != SQLITE_OK ||
                step_for_id(restore_stmt, id, 0) != SQLITE_OK) {
                fprintf(stderr, "sqlite statement failed: %s\n",
                        sqlite3_errmsg(database));
                batch_error = 1;
                break;
            }
        }
        sqlite3_reset(select_stmt);
        sqlite3_reset(update_stmt);
        sqlite3_reset(restore_stmt);
    }
}

static int setup_database(void) {
    sqlite3_stmt *insert_stmt = NULL;
    int id;
    if (sqlite3_open(":memory:", &database) != SQLITE_OK) return SQLITE_ERROR;
    if (execute("PRAGMA journal_mode=OFF") != SQLITE_OK ||
        execute("PRAGMA synchronous=OFF") != SQLITE_OK ||
        execute("CREATE TABLE items("
                "id INTEGER PRIMARY KEY, value INTEGER NOT NULL)") != SQLITE_OK ||
        execute("BEGIN") != SQLITE_OK) {
        return SQLITE_ERROR;
    }
    if (sqlite3_prepare_v2(
            database, "INSERT INTO items VALUES(?1, ?2)", -1,
            &insert_stmt, NULL) != SQLITE_OK) {
        return SQLITE_ERROR;
    }
    for (id = 1; id <= ROW_COUNT; id++) {
        sqlite3_reset(insert_stmt);
        sqlite3_bind_int(insert_stmt, 1, id);
        sqlite3_bind_int(insert_stmt, 2, id);
        if (sqlite3_step(insert_stmt) != SQLITE_DONE) return SQLITE_ERROR;
    }
    sqlite3_finalize(insert_stmt);
    if (execute("COMMIT") != SQLITE_OK) return SQLITE_ERROR;
    if (sqlite3_prepare_v2(
            database, "SELECT value FROM items WHERE id=?1", -1,
            &select_stmt, NULL) != SQLITE_OK ||
        sqlite3_prepare_v2(
            database, "UPDATE items SET value=value+1 WHERE id=?1", -1,
            &update_stmt, NULL) != SQLITE_OK ||
        sqlite3_prepare_v2(
            database, "UPDATE items SET value=value-1 WHERE id=?1", -1,
            &restore_stmt, NULL) != SQLITE_OK) {
        return SQLITE_ERROR;
    }
    return SQLITE_OK;
}

static sqlite3_int64 checksum(void) {
    sqlite3_stmt *statement = NULL;
    sqlite3_int64 value = -1;
    if (sqlite3_prepare_v2(
            database, "SELECT sum(value) FROM items", -1,
            &statement, NULL) == SQLITE_OK &&
        sqlite3_step(statement) == SQLITE_ROW) {
        value = sqlite3_column_int64(statement, 0);
    }
    sqlite3_finalize(statement);
    return value;
}

int main(int argc, char **argv) {
    const sqlite3_int64 expected =
        (sqlite3_int64)ROW_COUNT * (ROW_COUNT + 1) / 2;
    bench_result result;

    if (setup_database() != SQLITE_OK) {
        fprintf(stderr, "sqlite setup failed: %s\n",
                database ? sqlite3_errmsg(database) : "open failed");
        return 1;
    }

    /* Validate one complete unit before any timed work. */
    sqlite_batch(1, NULL);
    sqlite3_int64 actual = checksum();
    if (batch_error || actual != expected) {
        fprintf(stderr,
                "sqlite validation failed: expected %lld, observed %lld\n",
                (long long)expected, (long long)actual);
        return 1;
    }
    printf("sqlite: checksum = %lld\n", (long long)expected);

    result = bench_run(sqlite_batch, NULL, argc, argv);
    if (batch_error) return 1;
    printf("sqlite: rate = %.2f iteration/s\n", result.rate);

    sqlite3_finalize(select_stmt);
    sqlite3_finalize(update_stmt);
    sqlite3_finalize(restore_stmt);
    sqlite3_close(database);
    return 0;
}
