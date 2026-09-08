"""Compile a restricted scalar PTX kernel to Metal during local model conversion."""

import re


def compile_kernel(source: str) -> str:
    class Parser:
        def __init__(self, source):
            self.param_base = 65536
            self.symbols = {"WARP_SZ": 32}
            self.shared = []
            parameter = re.search(r"\.param \.align 8 \.b8 (\w+)\[(\d+)\]", source)
            if parameter is None or int(parameter[2]) != 328:
                raise ValueError("Expected a 328-byte scalar-kernel parameter block")
            self.symbols[parameter[1]] = self.param_base
            for i, (symbol, size) in enumerate(
                re.findall(r"\.shared[^;]+ (\S+)\[(\d+)\];", source)
            ):
                base = 0x10000000 + i * 0x10000
                self.symbols[symbol] = base
                self.shared.append((symbol, base, int(size)))
            self.instructions = []
            self.labels = {}
            body = source[
                source.index("{", source.index(".visible .entry")) + 1 :
            ].split("\n.entry ")[0]
            for line in body.splitlines():
                line = line.split("//")[0].strip()
                if line.startswith("{"):
                    line = line[1:].strip()
                if line.endswith("}") and ";" in line:
                    line = line[:-1].strip()
                if not line or line[0] in ".}":
                    continue
                if line.endswith(":"):
                    self.labels[line[:-1]] = len(self.instructions)
                    continue
                if not line.endswith(";"):
                    continue
                line = line[:-1]
                pred = None
                if line.startswith("@"):
                    pred, line = line[1:].split(None, 1)
                op, _, args = line.partition(" ")
                self.instructions.append(
                    (op, re.findall(r"\{[^}]+\}|\[[^]]+\]|[^,\s]+", args), pred, 0)
                )

    parser = Parser(source)

    def name(x):
        return re.sub(r"[^\w]", "_", x.lstrip("%"))

    special = {
        "%ctaid.x": "threadgroup_position_in_grid.x",
        "%ctaid.y": "threadgroup_position_in_grid.y",
        "%ctaid.z": "threadgroup_position_in_grid.z",
        "%tid.x": "thread_position_in_threadgroup.x",
        "%tid.y": "thread_position_in_threadgroup.y",
        "%tid.z": "thread_position_in_threadgroup.z",
        "%laneid": "thread_index_in_simdgroup",
        "WARP_SZ": "32u",
    }

    def raw(x):
        if x in special:
            return special[x]
        if x in parser.symbols:
            return f"{parser.symbols[x]}ul"
        if x.startswith("%") or x in ["low", "high", "f", "r", "__$temp3"]:
            return name(x)
        if x.startswith("0f"):
            return f"0x{x[2:]}u"
        return str(int(x, 0) & 0xFFFFFFFFFFFFFFFF) + "ul"

    def value(x, dtype):
        v = raw(x)
        return {
            "f32": f"as_type<float>(uint({v}))",
            "f16": f"as_type<half>(ushort({v}))",
            "f16x2": f"as_type<half2>(uint({v}))",
            "s32": f"as_type<int>(uint({v}))",
            "s16": f"as_type<short>(ushort({v}))",
            "s64": f"as_type<long>(ulong({v}))",
            "u16": f"ushort({v})",
            "b16": f"ushort({v})",
            "u32": f"uint({v})",
            "b32": f"uint({v})",
        }.get(dtype, v)

    def assign(target, expr, dtype=None):
        if dtype == "f32":
            expr = f"as_type<uint>(float({expr}))"
        elif dtype == "f16":
            expr = f"as_type<ushort>(half({expr}))"
        elif dtype == "f16x2":
            expr = f"as_type<uint>(half2({expr}))"
        typ = (
            "ulong" if target.startswith("%rd") or target in ["%SP", "%SPL"] else "uint"
        )
        return f"{name(target)}={typ}({expr});"

    def address(x):
        x = x.strip("[]")
        a, sep, b = x.partition("+")
        return f"({raw(a)}+({b}))" if sep else raw(a)

    def reglist(x):
        return [y.strip() for y in x.strip("{}").split(",")]

    header = []
    gdecl = "const device half* net, const device half* filter, const device uint* dims"
    gargs = "net, filter, dims"
    for bits in [16, 32, 64]:
        typ = {16: "ushort", 32: "uint", 64: "ulong"}[bits]
        cases = f"if(a>=4294967296ul && a+{bits // 8}<=4294967296ul+ulong(dims[4])*dims[5]*80) return *((const device {typ}*)((const device uchar*)net+a-4294967296ul));"
        cases += f"if(a>=8589934592ul && a+{bits // 8}<=8589934592ul+65536) return *((const device {typ}*)((const device uchar*)filter+a-8589934592ul));"
        header.append(f"{typ} global{bits}(ulong a,{gdecl}){{" + cases + "return 0;}")
    shared = parser.shared
    sdecl = ", ".join(f"threadgroup uchar* {name(symbol)}" for symbol, _, _ in shared)
    sargs = ", ".join(name(symbol) for symbol, _, _ in shared)
    for bits in [16, 32]:
        typ = {16: "ushort", 32: "uint"}[bits]
        read = []
        write = []
        for symbol, base, size in shared:
            pointer = f"((threadgroup {typ}*)({name(symbol)}+a-{base}ul))"
            condition = f"a>={base}ul && a+{bits // 8}<={base + size}ul"
            read.append(f"if({condition}) return *{pointer};")
            write.append(f"if({condition}){{*{pointer}=v;return;}}")
        header.append(
            f"{typ} shared{bits}(ulong a,{sdecl}){{" + "".join(read) + "return 0;}"
        )
        header.append(
            f"void store{bits}(ulong a,{typ} v,{sdecl}){{" + "".join(write) + "}"
        )
    tdecl = "const device float* color, const device float* previousColor, const device float* previousLuma, const device float* motion, const device float* depth, const device uint* dims"
    targs = "color, previousColor, previousLuma, motion, depth, dims"
    header.append(
        "float4 pointRead(const device float* t,int2 p,int w,int h,int c){p=clamp(p,int2(0),int2(w,h)-1);uint i=(p.y*w+p.x)*c;return c==4?float4(t[i],t[i+1],t[i+2],t[i+3]):float4(t[i],0,0,0);}"
    )
    header.append(
        "float4 linearRead(const device float* t,float2 p,int w,int h,int c){p-=.5f;int2 i=int2(floor(p));float2 f=rint((p-float2(i))*256.f)/256.f;return mix(mix(pointRead(t,i,w,h,c),pointRead(t,i+int2(1,0),w,h,c),f.x),mix(pointRead(t,i+int2(0,1),w,h,c),pointRead(t,i+1,w,h,c),f.x),f.y);}"
    )
    header.append(
        "float4 gatherRead(const device float* t,float2 p,int w,int h,int c){int2 i=int2(floor(p-.5f));return float4(pointRead(t,i+int2(0,1),w,h,c).x,pointRead(t,i+1,w,h,c).x,pointRead(t,i+int2(1,0),w,h,c).x,pointRead(t,i,w,h,c).x);}"
    )
    cases = []
    for handle, key, w, h, c, filtered, normalized in [
        (1, "color", "dims[0]", "dims[1]", 4, False, False),
        (2, "previousColor", "dims[6]", "dims[7]", 4, True, True),
        (3, "previousLuma", "dims[6]*2", "dims[7]*2", 1, True, True),
        (4, "motion", "dims[0]", "dims[1]", 4, False, False),
        (5, "depth", "dims[0]", "dims[1]", 4, False, False),
    ]:
        cases.append(
            f"case {handle}:{{float2 p=integer?coordinate:coordinate*float2({w if normalized else 1},{h if normalized else 1});if(gather)return gatherRead({key},p,{w},{h},{c});return "
            + (
                f"integer?pointRead({key},int2(p),{w},{h},{c}):linearRead({key},p,{w},{h},{c});"
                if filtered
                else f"pointRead({key},int2(floor(p)),{w},{h},{c});"
            )
            + "}"
        )
    header.append(
        f"float4 textureRead(ulong handle,float2 coordinate,bool integer,bool gather,{tdecl}){{switch(handle){{"
        + "".join(cases)
        + "default:return float4(0);}}"
    )
    output_names = ["rgb", "history", "luma", "hidden"]
    odecl = "device float* rgb, device half* history, device half* luma, device half* hidden, const device uint* dims"
    oargs = ", ".join(output_names) + ", dims"
    cases = []
    for handle, key, w, h, c, typ in [
        (6, "rgb", "dims[2]", "dims[3]", 4, "float"),
        (7, "history", "dims[6]", "dims[7]", 4, "half"),
        (8, "luma", "dims[6]*2", "dims[7]*2", 1, "half"),
        (9, "hidden", "dims[4]", "dims[5]", 4, "half"),
    ]:
        pitch = f"({w})*{c * (4 if typ == 'float' else 2)}"
        cases.append(
            f"case {handle}:if(byteMode){{if(p.y>=0&&p.y<int({h})&&p.x>=0&&p.x<=int({pitch})-4)*((device uint*)((device uchar*){key}+p.y*({pitch})+p.x))=v.x;}}else{{if(all(p>=0)&&all(p<int2({w},{h})))((device {typ}4*){key})[p.y*({w})+p.x]={typ}4(as_type<float4>(v));}}break;"
        )
    header.append(
        f"void surfaceWrite(ulong handle,int2 p,uint4 v,bool byteMode,{odecl}){{switch(handle){{"
        + "".join(cases)
        + "}}"
    )
    header.append(
        "uint bitInsert(uint a,uint b,uint position,uint count){uint mask=(0xffffffffu>>(32u-count))<<position;return (b&~mask)|((a<<position)&mask);}"
    )

    def instruction(op, a):
        base = op.split(".")[0]
        dtype = op.split(".")[-1]
        if base == "bra":
            return "goto " + name(a[0]) + ";"
        if base == "ret":
            return "return;"
        if base == "bar":
            return "threadgroup_barrier(mem_flags::mem_threadgroup);"
        if base in ["ld", "st"]:
            width = int(re.search(r"(\d+)$", dtype)[1])
            group = re.search(r"\.v([24])\.", op)
            count = int(group[1]) if group else 1
            addr = address(a[1] if base == "ld" else a[0])
            names = reglist(a[0] if base == "ld" else a[1])
            lines = []
            for i in range(count):
                current = f"({addr}+{i * width // 8})"
                typ = {8: "uchar", 16: "ushort", 32: "uint", 64: "ulong"}[width]
                if base == "ld":
                    if ".param." in op:
                        expr = f"*((const device {typ}*)(params+{current}-{parser.param_base}ul))"
                    elif ".shared." in op:
                        expr = f"shared{width}({current},{sargs})"
                    elif ".global." in op:
                        expr = f"global{width}({current},{gargs})"
                    else:
                        raise ValueError(op)
                    lines.append(assign(names[i], expr))
                else:
                    lines.append(
                        f"store{width}({current},{typ}({raw(names[i])}),{sargs});"
                    )
            return "".join(lines)
        if base in ["tex", "tld4"]:
            operands = re.findall(r"%\w+", a[1])
            ctype = "s32" if op.endswith("s32") else "f32"
            coordinates = ",".join(value(v, ctype) for v in operands[1:])
            temp = f"textureRead({raw(operands[0])},float2({coordinates}),{str(ctype == 's32').lower()},{str(base == 'tld4').lower()},{targs})"
            return (
                "{float4 sample="
                + temp
                + ";"
                + "".join(
                    assign(v, "sample." + "xyzw"[i], "f32")
                    for i, v in enumerate(reglist(a[0]))
                )
                + "}"
            )
        if base == "sust":
            operands = re.findall(r"%\w+", a[0])
            coordinates = ",".join(value(v, "s32") for v in operands[1:])
            vals = [raw(v) for v in reglist(a[1])]
            vals += ["0"] * (4 - len(vals))
            return f"surfaceWrite({raw(operands[0])},int2({coordinates}),uint4({','.join(vals)}),{str('.b.' in op).lower()},{oargs});"
        if base == "mov":
            if a[0].startswith("{"):
                names = reglist(a[0])
                bits = int(dtype[1:]) // len(names)
                return "".join(
                    assign(v, f"({raw(a[1])}>>{i * bits})&{(1 << bits) - 1}ul")
                    for i, v in enumerate(names)
                )
            if a[1].startswith("{"):
                names = reglist(a[1])
                bits = int(dtype[1:]) // len(names)
                return assign(
                    a[0],
                    "|".join(
                        f"(ulong({raw(v)})<<{i * bits})" for i, v in enumerate(names)
                    ),
                )
            return assign(a[0], raw(a[1]))
        if base == "cvta":
            return assign(a[0], raw(a[1]))
        if base == "cvt":
            dest, src = op.split(".")[-2:]
            v = value(a[1], src)
            if dest == "f16x2":
                return assign(a[0], f"half2({value(a[2], src)},{v})", dest)
            if any(vv in op for vv in [".rzi.", ".rni.", ".rmi.", ".rpi."]):
                fn = (
                    "trunc"
                    if ".rzi." in op
                    else "rint"
                    if ".rni." in op
                    else "floor"
                    if ".rmi." in op
                    else "ceil"
                )
                v = f"{fn}(float({v}))"
            if ".sat." in op:
                v = f"clamp({v},0.f,1.f)"
            if not dest.startswith("f"):
                v = f"{'int' if dest.startswith('s') else 'uint'}({v})"
            return assign(a[0], v, dest if dest.startswith("f") else None)
        if base == "shfl":
            names = a[0].split("|")
            stmt = assign(
                names[0], f"simd_shuffle_xor(uint({raw(a[1])}),ushort({raw(a[2])}))"
            )
            return stmt + (assign(names[1], "1") if len(names) > 1 else "")
        if base in ["setp", "set"]:
            cmp = op.split(".")[1]
            cmpop = {
                "gt": ">",
                "ge": ">=",
                "lt": "<",
                "le": "<=",
                "eq": "==",
                "ne": "!=",
            }[cmp.rstrip("u")]
            v = f"({value(a[1], dtype)}{cmpop}{value(a[2], dtype)})"
            if cmp.endswith("u"):
                v = f"({v}||isnan({value(a[1], dtype)})||isnan({value(a[2], dtype)}))"
            if base == "set":
                return (
                    "{bool2 match="
                    + v
                    + ";"
                    + assign(a[0], "(match.x?65535u:0u)|(match.y?0xffff0000u:0u)")
                    + "}"
                )
            return assign(a[0], v)
        if base == "selp":
            return assign(a[0], f"({raw(a[3])}?{raw(a[1])}:{raw(a[2])})")
        values = [value(v, dtype) for v in a[1:]]
        x = values[0]
        if base in ["add", "sub", "mul", "div", "and", "or", "xor", "shl", "shr"]:
            operator = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "and": "&",
                "or": "|",
                "xor": "^",
                "shl": "<<",
                "shr": ">>",
            }[base]
            if base == "mul" and ".wide." in op:
                values = [f"ulong({v})" for v in values]
            v = f"({values[0]} {operator} {values[1]})"
        elif base == "fma":
            v = f"fma({','.join(values)})"
        elif base == "mad":
            v = f"({values[0]}*{values[1]}+{values[2]})"
        elif base in ["min", "max"]:
            v = f"{base}({values[0]},{values[1]})"
        elif base == "not":
            v = f"({x}^1u)" if dtype == "pred" else f"(~{x})"
        elif base == "neg":
            v = f"(-{x})"
        elif base in ["abs", "rsqrt", "ex2", "lg2"]:
            v = f"{ {'ex2': 'exp2', 'lg2': 'log2'}.get(base, base) }({x})"
        elif base == "rcp":
            v = f"(1.f/{x})"
        elif base == "bfi":
            v = f"bitInsert({','.join('uint(' + v + ')' for v in values)})"
        else:
            raise NotImplementedError(op)
        if ".sat." in op:
            v = f"clamp({v},{'half2' if dtype == 'f16x2' else 'half' if dtype == 'f16' else 'float'}(0),{'half2' if dtype == 'f16x2' else 'half' if dtype == 'f16' else 'float'}(1))"
        return assign(a[0], v, dtype if dtype.startswith("f") else None)

    registers = (
        set(re.findall(r"%[\w$]+", source))
        - set(special)
        - {"%r", "%f", "%rs", "%rd", "%p"}
    )
    registers |= {"low", "high", "f", "r", "__$temp3"}
    declarations = []
    for reg in sorted(registers):
        typ = "ulong" if reg.startswith("%rd") or reg in ["%SP", "%SPL"] else "uint"
        declarations.append(f"{typ} {name(reg)}=0;")
    declarations += [
        f"threadgroup uchar {name(symbol)}[{size}];" for symbol, base, size in shared
    ]
    boundaries = {0, *parser.labels.values(), len(parser.instructions)}
    for pc, (op, _, _, _) in enumerate(parser.instructions):
        if op.split(".")[0] in ["bra", "ret"]:
            boundaries.add(pc + 1)
    starts = sorted(boundaries)
    blocks = {a: parser.instructions[a:b] for a, b in zip(starts, starts[1:])}
    edges = {}
    for start, body in blocks.items():
        op, args, pred, _ = body[-1]
        end = start + len(body)
        if op.split(".")[0] == "bra":
            edges[start] = [parser.labels[args[0]]] + ([end] if pred else [])
        elif op == "ret":
            edges[start] = []
        else:
            edges[start] = [end] if end < len(parser.instructions) else []
    index = {}
    low = {}
    stack = []
    onstack = set()
    components = []

    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        onstack.add(node)
        for to in edges[node]:
            if to not in index:
                visit(to)
                low[node] = min(low[node], low[to])
            elif to in onstack:
                low[node] = min(low[node], index[to])
        if low[node] == index[node]:
            component = []
            while True:
                n = stack.pop()
                onstack.remove(n)
                component.append(n)
                if n == node:
                    break
            components.append(sorted(component))

    visit(0)
    components.reverse()
    lines = ["uint control=0;"]
    for component in components:
        cyclic = len(component) > 1 or component[0] in edges[component[0]]
        if cyclic:
            lines.append(
                "while(" + ("||".join(f"control=={pc}" for pc in component)) + "){"
            )
        for pc in component:
            lines.append(f"if(control=={pc}){{")
            body = blocks[pc]
            for j, (op, args, pred, _) in enumerate(body):
                base = op.split(".")[0]
                condition = (
                    f"{'!' if pred and pred.startswith('!') else ''}{raw(pred.lstrip('!'))}"
                    if pred
                    else None
                )
                if base == "bra":
                    target = parser.labels[args[0]]
                    lines.append(
                        f"control={condition}?{target}:{pc + len(body)};"
                        if pred
                        else f"control={target};"
                    )
                elif base == "ret":
                    lines.append("return;")
                else:
                    stmt = instruction(op, args)
                    if pred:
                        stmt = f"if({condition}){{{stmt}}}"
                    lines.append(stmt)
                    if j == len(body) - 1:
                        lines.append(f"control={pc + len(body)};")
            lines.append("}")
        if cyclic:
            lines.append("}")
    return "\n".join(header) + "\n// END HEADER\n" + "\n".join(declarations + lines)
