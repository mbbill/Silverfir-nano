-- JSON Parse/Encode Benchmark using lunajson (inlined, pure Lua)
-- Tests real-world JSON processing: decode + encode round-trip

---------------------------------------------------------------------------
-- Inlined lunajson decoder (from github.com/grafi-tt/lunajson)
---------------------------------------------------------------------------
local function newdecoder()
    local setmetatable, tonumber, tostring =
          setmetatable, tonumber, tostring
    local floor, inf =
          math.floor, math.huge
    local mininteger, tointeger =
          math.mininteger or nil, math.tointeger or nil
    local byte, char, find, gsub, match, sub =
          string.byte, string.char, string.find, string.gsub, string.match, string.sub

    local function _decode_error(pos, errmsg)
        error("parse error at " .. pos .. ": " .. errmsg, 2)
    end

    local f_str_ctrl_pat = '[\0-\31]'

    local json, pos, nullv, arraylen, rec_depth
    local dispatcher, f

    local function decode_error(errmsg)
        return _decode_error(pos, errmsg)
    end

    local function f_err() decode_error('invalid value') end

    local function f_nul()
        if sub(json, pos, pos+2) == 'ull' then pos = pos+3; return nullv end
        decode_error('invalid value')
    end

    local function f_fls()
        if sub(json, pos, pos+3) == 'alse' then pos = pos+4; return false end
        decode_error('invalid value')
    end

    local function f_tru()
        if sub(json, pos, pos+2) == 'rue' then pos = pos+3; return true end
        decode_error('invalid value')
    end

    local radixmark = match(tostring(0.5), '[^0-9]')
    local fixedtonumber = tonumber
    if radixmark ~= '.' then
        if find(radixmark, '%W') then radixmark = '%' .. radixmark end
        fixedtonumber = function(s) return tonumber(gsub(s, '.', radixmark)) end
    end

    local function number_error() return decode_error('invalid number') end

    local function f_zro(mns)
        local num, c = match(json, '^(%.?[0-9]*)([-+.A-Za-z]?)', pos)
        if num == '' then
            if c == '' then return mns and -0.0 or 0 end
            if c == 'e' or c == 'E' then
                num, c = match(json, '^([^eE]*[eE][-+]?[0-9]+)([-+.A-Za-z]?)', pos)
                if c == '' then pos = pos + #num; return mns and -0.0 or 0.0 end
            end
            number_error()
        end
        if byte(num) ~= 0x2E or byte(num, -1) == 0x2E then number_error() end
        if c ~= '' then
            if c == 'e' or c == 'E' then
                num, c = match(json, '^([^eE]*[eE][-+]?[0-9]+)([-+.A-Za-z]?)', pos)
            end
            if c ~= '' then number_error() end
        end
        pos = pos + #num
        c = fixedtonumber(num)
        if mns then c = -c end
        return c
    end

    local function f_num(mns)
        pos = pos-1
        local num, c = match(json, '^([0-9]+%.?[0-9]*)([-+.A-Za-z]?)', pos)
        if byte(num, -1) == 0x2E then number_error() end
        if c ~= '' then
            if c ~= 'e' and c ~= 'E' then number_error() end
            num, c = match(json, '^([^eE]*[eE][-+]?[0-9]+)([-+.A-Za-z]?)', pos)
            if not num or c ~= '' then number_error() end
        end
        pos = pos + #num
        c = fixedtonumber(num)
        if mns then
            c = -c
            if c == mininteger and not find(num, '[^0-9]') then c = mininteger end
        end
        return c
    end

    local function f_mns()
        local c = byte(json, pos)
        if c then
            pos = pos+1
            if c > 0x30 then if c < 0x3A then return f_num(true) end
            else if c > 0x2F then return f_zro(true) end end
        end
        decode_error('invalid number')
    end

    local f_str_hextbl = {
        0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
        0x8, 0x9, inf, inf, inf, inf, inf, inf,
        inf, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, inf,
        inf, inf, inf, inf, inf, inf, inf, inf,
        inf, inf, inf, inf, inf, inf, inf, inf,
        inf, inf, inf, inf, inf, inf, inf, inf,
        inf, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF,
        __index = function() return inf end
    }
    setmetatable(f_str_hextbl, f_str_hextbl)

    local f_str_escapetbl = {
        ['"']  = '"', ['\\'] = '\\', ['/']  = '/',
        ['b']  = '\b', ['f']  = '\f', ['n']  = '\n',
        ['r']  = '\r', ['t']  = '\t',
        __index = function() decode_error("invalid escape sequence") end
    }
    setmetatable(f_str_escapetbl, f_str_escapetbl)

    local function surrogate_first_error()
        return decode_error("1st surrogate pair byte not continued by 2nd")
    end

    local f_str_surrogate_prev = 0
    local function f_str_subst(ch, ucode)
        if ch == 'u' then
            local c1, c2, c3, c4, rest = byte(ucode, 1, 5)
            ucode = f_str_hextbl[c1-47] * 0x1000 +
                    f_str_hextbl[c2-47] * 0x100 +
                    f_str_hextbl[c3-47] * 0x10 +
                    f_str_hextbl[c4-47]
            if ucode ~= inf then
                if ucode < 0x80 then
                    if rest then return char(ucode, rest) end
                    return char(ucode)
                elseif ucode < 0x800 then
                    c1 = floor(ucode / 0x40)
                    c2 = ucode - c1 * 0x40
                    c1 = c1 + 0xC0; c2 = c2 + 0x80
                    if rest then return char(c1, c2, rest) end
                    return char(c1, c2)
                elseif ucode < 0xD800 or 0xE000 <= ucode then
                    c1 = floor(ucode / 0x1000)
                    ucode = ucode - c1 * 0x1000
                    c2 = floor(ucode / 0x40)
                    c3 = ucode - c2 * 0x40
                    c1 = c1 + 0xE0; c2 = c2 + 0x80; c3 = c3 + 0x80
                    if rest then return char(c1, c2, c3, rest) end
                    return char(c1, c2, c3)
                elseif 0xD800 <= ucode and ucode < 0xDC00 then
                    if f_str_surrogate_prev == 0 then
                        f_str_surrogate_prev = ucode
                        if not rest then return '' end
                        surrogate_first_error()
                    end
                    f_str_surrogate_prev = 0
                    surrogate_first_error()
                else
                    if f_str_surrogate_prev ~= 0 then
                        ucode = 0x10000 +
                                (f_str_surrogate_prev - 0xD800) * 0x400 +
                                (ucode - 0xDC00)
                        f_str_surrogate_prev = 0
                        c1 = floor(ucode / 0x40000)
                        ucode = ucode - c1 * 0x40000
                        c2 = floor(ucode / 0x1000)
                        ucode = ucode - c2 * 0x1000
                        c3 = floor(ucode / 0x40)
                        c4 = ucode - c3 * 0x40
                        c1 = c1 + 0xF0; c2 = c2 + 0x80; c3 = c3 + 0x80; c4 = c4 + 0x80
                        if rest then return char(c1, c2, c3, c4, rest) end
                        return char(c1, c2, c3, c4)
                    end
                    decode_error("2nd surrogate pair byte appeared without 1st")
                end
            end
            decode_error("invalid unicode codepoint literal")
        end
        if f_str_surrogate_prev ~= 0 then
            f_str_surrogate_prev = 0
            surrogate_first_error()
        end
        return f_str_escapetbl[ch] .. ucode
    end

    local f_str_keycache = setmetatable({}, {__mode="v"})

    local function f_str(iskey)
        local newpos = pos
        local tmppos, c1, c2
        repeat
            newpos = find(json, '"', newpos, true)
            if not newpos then decode_error("unterminated string") end
            tmppos = newpos-1; newpos = newpos+1
            c1, c2 = byte(json, tmppos-1, tmppos)
            if c2 == 0x5C and c1 == 0x5C then
                repeat tmppos = tmppos-2; c1, c2 = byte(json, tmppos-1, tmppos)
                until c2 ~= 0x5C or c1 ~= 0x5C
                tmppos = newpos-2
            end
        until c2 ~= 0x5C
        local str = sub(json, pos, tmppos)
        pos = newpos
        if iskey then
            tmppos = f_str_keycache[str]
            if tmppos then return tmppos end
            tmppos = str
        end
        if find(str, f_str_ctrl_pat) then decode_error("unescaped control string") end
        if find(str, '\\', 1, true) then
            str = gsub(str, '\\(.)([^\\]?[^\\]?[^\\]?[^\\]?[^\\]?)', f_str_subst)
            if f_str_surrogate_prev ~= 0 then
                f_str_surrogate_prev = 0
                decode_error("1st surrogate pair byte not continued by 2nd")
            end
        end
        if iskey then f_str_keycache[tmppos] = str end
        return str
    end

    local function f_ary()
        rec_depth = rec_depth + 1
        if rec_depth > 1000 then decode_error('too deeply nested json (> 1000)') end
        local ary = {}
        pos = match(json, '^[ \n\r\t]*()', pos)
        local i = 0
        if byte(json, pos) == 0x5D then
            pos = pos+1
        else
            local newpos = pos
            repeat
                i = i+1
                f = dispatcher[byte(json,newpos)]
                pos = newpos+1
                ary[i] = f()
                newpos = match(json, '^[ \n\r\t]*,[ \n\r\t]*()', pos)
            until not newpos
            newpos = match(json, '^[ \n\r\t]*%]()', pos)
            if not newpos then decode_error("no closing bracket of an array") end
            pos = newpos
        end
        if arraylen then ary[0] = i end
        rec_depth = rec_depth - 1
        return ary
    end

    local function f_obj()
        rec_depth = rec_depth + 1
        if rec_depth > 1000 then decode_error('too deeply nested json (> 1000)') end
        local obj = {}
        pos = match(json, '^[ \n\r\t]*()', pos)
        if byte(json, pos) == 0x7D then
            pos = pos+1
        else
            local newpos = pos
            repeat
                if byte(json, newpos) ~= 0x22 then decode_error("not key") end
                pos = newpos+1
                local key = f_str(true)
                f = f_err
                local c1, c2, c3 = byte(json, pos, pos+3)
                if c1 == 0x3A then
                    if c2 ~= 0x20 then f = dispatcher[c2]; newpos = pos+2
                    else f = dispatcher[c3]; newpos = pos+3 end
                end
                if f == f_err then
                    newpos = match(json, '^[ \n\r\t]*:[ \n\r\t]*()', pos)
                    if not newpos then decode_error("no colon after a key") end
                    f = dispatcher[byte(json, newpos)]
                    newpos = newpos+1
                end
                pos = newpos
                obj[key] = f()
                newpos = match(json, '^[ \n\r\t]*,[ \n\r\t]*()', pos)
            until not newpos
            newpos = match(json, '^[ \n\r\t]*}()', pos)
            if not newpos then decode_error("no closing bracket of an object") end
            pos = newpos
        end
        rec_depth = rec_depth - 1
        return obj
    end

    dispatcher = { [0] =
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_str, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_mns, f_err, f_err,
        f_zro, f_num, f_num, f_num, f_num, f_num, f_num, f_num,
        f_num, f_num, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_ary, f_err, f_err, f_err, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_fls, f_err,
        f_err, f_err, f_err, f_err, f_err, f_err, f_nul, f_err,
        f_err, f_err, f_err, f_err, f_tru, f_err, f_err, f_err,
        f_err, f_err, f_err, f_obj, f_err, f_err, f_err, f_err,
        __index = function() decode_error("unexpected termination") end
    }
    setmetatable(dispatcher, dispatcher)

    local function decode(json_, pos_, nullv_, arraylen_)
        json, pos, nullv, arraylen = json_, pos_, nullv_, arraylen_
        rec_depth = 0
        pos = match(json, '^[ \n\r\t]*()', pos)
        f = dispatcher[byte(json, pos)]
        pos = pos+1
        local v = f()
        if pos_ then return v, pos
        else
            f, pos = find(json, '^[ \n\r\t]*', pos)
            if pos ~= #json then decode_error('json ended') end
            return v
        end
    end
    return decode
end

---------------------------------------------------------------------------
-- Inlined lunajson encoder
---------------------------------------------------------------------------
local function newencoder()
    local error = error
    local byte, find, format, gsub, match = string.byte, string.find, string.format, string.gsub, string.match
    local concat = table.concat
    local tostring = tostring
    local pairs, type = pairs, type
    local setmetatable = setmetatable
    local huge, tiny = 1/0, -1/0

    local v, nullv
    local i, builder, visited

    local function f_tostring(v)
        builder[i] = tostring(v); i = i+1
    end

    local radixmark = match(tostring(0.5), '[^0-9]')
    local delimmark = match(tostring(12345.12345), '[^0-9' .. radixmark .. ']')
    if radixmark == '.' then radixmark = nil end

    local radixordelim
    if radixmark or delimmark then
        radixordelim = true
        if radixmark and find(radixmark, '%W') then radixmark = '%' .. radixmark end
        if delimmark and find(delimmark, '%W') then delimmark = '%' .. delimmark end
    end

    local f_number = function(n)
        if tiny < n and n < huge then
            local s = format("%.17g", n)
            if radixordelim then
                if delimmark then s = gsub(s, delimmark, '') end
                if radixmark then s = gsub(s, radixmark, '.') end
            end
            builder[i] = s; i = i+1; return
        end
        error('invalid number')
    end

    local doencode

    local f_string_esc_pat = '[\0-\31"\\]'
    local f_string_subst = {
        ['"'] = '\\"', ['\\'] = '\\\\',
        ['\b'] = '\\b', ['\f'] = '\\f', ['\n'] = '\\n',
        ['\r'] = '\\r', ['\t'] = '\\t',
        __index = function(_, c) return format('\\u00%02X', byte(c)) end
    }
    setmetatable(f_string_subst, f_string_subst)

    local function f_string(s)
        builder[i] = '"'
        if find(s, f_string_esc_pat) then s = gsub(s, f_string_esc_pat, f_string_subst) end
        builder[i+1] = s; builder[i+2] = '"'; i = i+3
    end

    local function f_table(o)
        if visited[o] then error("loop detected") end
        visited[o] = true
        local tmp = o[0]
        if type(tmp) == 'number' then
            builder[i] = '['; i = i+1
            for j = 1, tmp do doencode(o[j]); builder[i] = ','; i = i+1 end
            if tmp > 0 then i = i-1 end
            builder[i] = ']'
        else
            tmp = o[1]
            if tmp ~= nil then
                builder[i] = '['; i = i+1
                local j = 2
                repeat
                    doencode(tmp); tmp = o[j]
                    if tmp == nil then break end
                    j = j+1; builder[i] = ','; i = i+1
                until false
                builder[i] = ']'
            else
                builder[i] = '{'; i = i+1
                local tmp = i
                for k, v in pairs(o) do
                    if type(k) ~= 'string' then error("non-string key") end
                    f_string(k); builder[i] = ':'; i = i+1
                    doencode(v); builder[i] = ','; i = i+1
                end
                if i > tmp then i = i-1 end
                builder[i] = '}'
            end
        end
        i = i+1; visited[o] = nil
    end

    local dispatcher = {
        boolean = f_tostring, number = f_number,
        string = f_string, table = f_table,
        __index = function() error("invalid type value") end
    }
    setmetatable(dispatcher, dispatcher)

    function doencode(v)
        if v == nullv then builder[i] = 'null'; i = i+1; return end
        return dispatcher[type(v)](v)
    end

    local function encode(v_, nullv_)
        v, nullv = v_, nullv_
        i, builder, visited = 1, {}, {}
        doencode(v)
        return concat(builder)
    end
    return encode
end

---------------------------------------------------------------------------
-- Generate a realistic JSON dataset (simulates API response / config data)
---------------------------------------------------------------------------
local function generate_data(n_users)
    local users = {}
    local roles = {"admin", "editor", "viewer", "moderator", "contributor"}
    local cities = {"New York", "London", "Tokyo", "Berlin", "Sydney",
                    "Paris", "Toronto", "Mumbai", "Seoul", "Dubai"}
    local tags_pool = {"lua", "wasm", "systems", "web", "backend", "frontend",
                       "database", "networking", "security", "performance",
                       "testing", "devops", "mobile", "ai", "graphics"}

    for i = 1, n_users do
        local tags = {}
        -- each user gets 3-5 tags
        local ntags = 3 + (i % 3)
        for j = 1, ntags do
            tags[j] = tags_pool[((i * 7 + j * 3) % #tags_pool) + 1]
        end

        local scores = {}
        for j = 1, 5 do
            scores[j] = ((i * 17 + j * 31) % 1000) / 10.0
        end

        users[i] = {
            id = i,
            username = "user_" .. tostring(i),
            email = "user" .. tostring(i) .. "@example.com",
            active = (i % 3 ~= 0),
            role = roles[(i % #roles) + 1],
            profile = {
                city = cities[(i % #cities) + 1],
                age = 20 + (i % 45),
                bio = "This is the bio for user number " .. tostring(i) ..
                      ". They enjoy programming and open source software.",
            },
            tags = tags,
            scores = scores,
            metadata = {
                created_at = "2024-01-" .. string.format("%02d", (i % 28) + 1) .. "T12:00:00Z",
                login_count = i * 42 % 999,
                preferences = {
                    theme = (i % 2 == 0) and "dark" or "light",
                    notifications = (i % 4 ~= 0),
                    language = (i % 3 == 0) and "ja" or ((i % 3 == 1) and "en" or "de"),
                }
            }
        }
    end

    return {
        api_version = "2.1.0",
        total_count = n_users,
        page = 1,
        per_page = n_users,
        users = users,
    }
end

---------------------------------------------------------------------------
-- Benchmark (self-regulating, like CoreMark)
---------------------------------------------------------------------------
-- The time budget is the LAST argument (see bench.lua); an optional N_USERS
-- may precede it.
local TIME_BUDGET = (arg and #arg > 0 and tonumber(arg[#arg])) or 2

-- Shared timer. Some WASI runtimes expose only whole-second os.time();
-- bench.run aligns to its ticks and samples in small chunks so their reported
-- integer-second target has only about one chunk of endpoint overshoot.
local bench = dofile("bench.lua")

local decode = newdecoder()
local encode = newencoder()

local N_USERS = (arg and #arg >= 2 and tonumber(arg[1])) or 200

print(string.format("Generating dataset: %d users...", N_USERS))
local data = generate_data(N_USERS)

local json_str = encode(data)
print(string.format("JSON size: %d bytes (%.1f KB)", #json_str, #json_str / 1024))
local validation = decode(json_str, 1)
assert(validation.total_count == N_USERS)
assert(#validation.users == N_USERS)
assert(validation.users[1].id == data.users[1].id)
assert(validation.users[1].username == data.users[1].username)
print("JSON roundtrip validates")

-- Run round-trips until time budget is exhausted.
-- Check time every `batch` iterations to amortize gettime() overhead.
print(string.format("Target time:   %.1fs\n", TIME_BUDGET))

local bytes_per_roundtrip = 0
local function batch(n)
    for _ = 1, n do
        local obj = decode(json_str, 1)
        local out = encode(obj)
        bytes_per_roundtrip = #json_str + #out
    end
end

local _, iters_done, elapsed = bench.run(batch, TIME_BUDGET)
local total_bytes = iters_done * bytes_per_roundtrip

print("=== JSON Benchmark Results ===")
print(string.format("Round-trips:    %d", iters_done))
print(string.format("JSON size:      %.1f KB", #json_str / 1024))
print(string.format("Time:           %.3f s", elapsed))
print(string.format("Throughput:     %.0f KB/s", total_bytes / 1024 / elapsed))
print(string.format("Round-trips/s:  %.1f", iters_done / elapsed))
print(string.format("Score:          %.0f", total_bytes / 1024 / elapsed))
