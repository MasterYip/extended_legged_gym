# Interactive web viewer

This is the self-contained ENV-VIS-015 example. It includes Modes A/B/C,
Refined/Classic styling, Blender-like 3D navigation, top-down pan/zoom, and a
Compute selector for Live or On-release derived computation.

From the `el4090_envelope` project directory:

```bash
PYTHONPATH=src python examples/web_viewer/envelope_server.py --port 8766
```

Then open `http://127.0.0.1:8766/`. Use a port other than an existing deployed
viewer. The server binds only to `127.0.0.1`. Override the robot asset with
`EL4090_URDF=/absolute/path/to/el_4090.urdf` if it is not in the sibling
`legged_gym/resources/robots/el_4090/urdf` tree.

## Migrated source identities

The package-adapted sources have these SHA-256 identities:

| File | SHA-256 |
|---|---|
| `index.html` | `506a0d27af540b74bca79714a106d0326d4f815bb8e7fb1ec69e94a9fe5ea414` |
| `styles.css` | `311583718fa89c97baa2423621013832d6a9be7bd74c774954bdda21e774ba40` |
| `app.js` | `722b8e4fc2e90fade0295b5759339d31e903c0142bc88bbd03fd89b0caefd6ac` |
| `envelope_server.py` | `6226d28f6312ee56e3ad51339d6c01c36844fc76ec22572bf95635425d3016a8` |

The three frontend files are byte-identical to the migrated ENV-VIS-015
sources. `envelope_server.py` differs only in package/path wiring and its
module-level documentation; endpoint and visualization behavior is preserved.
