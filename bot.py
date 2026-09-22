import os
import io
import random
import asyncio
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Draw

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise SystemExit("Missing DISCORD_TOKEN in environment or .env file")

DEFAULT_SEEDS = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # acetaminophen
    "CCO",                             # ethanol
    "c1ccccc1",                        # benzene
    "CC(=O)O",                         # acetic acid
    "c1ccc2c(c1)ccc1ccccc12",         # anthracene
    "CC(C)NCC(O)",                     # isoproterenol-like
    "O=C(O)c1ccccc1O",                # salicylic acid
    "CC12CCC3C(CCC4CC(=O)CCC34C)C1CCC2O",  # testosterone
    "c1ccc(-c2ccccc2)cc1",            # biphenyl
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "c1ccc2[nH]ccc2c1",               # indole
    "C1CCCCC1",                        # cyclohexane
    "CC(C)(C)O",                       # t-butanol
    "O=c1[nH]c(=O)[nH]c(=O)[nH]1",   # uracil
    "c1ccncc1",                        # pyridine
    "CC=O",                            # acetaldehyde
    "O=C(O)CC(=O)O",                   # succinic acid
    "c1ccc2c(c1)cc1ccccc12",          # naphthalene
]

RANDOM_SEEDS = [
    "c1ccccc1", "C1CCCCC1", "c1ccncc1", "CC=O", "CCO",
    "CC(=O)O", "CC(C)O", "CCCC", "c1ccc2[nH]ccc2c1",
    "O=c1[nH]c(=O)[nH]c(=O)[nH]1", "O=C(O)CC(=O)O",
    "c1ccc2c(c1)cc1ccccc12", "CC(C)(C)O", "CC(=O)N",
    "c1ccc(-c2ccccc2)cc1", "C1CCNCC1", "O=C(O)c1ccccc1O",
    "c1ccc2c(c1)cco2", "C1COCCO1", "c1ccoc1",
]

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

_model = None


def get_model():
    global _model
    if _model is None:
        from RiemannMol.atom import load_model
        _model = load_model()
    return _model


def run_generate(seed: str, n_samples: int, std: float):
    from RiemannMol.atom import generate
    model = get_model()
    results = generate(seed, n_samples=n_samples, std=std, model=model)
    return results


def run_optimize(seed: str, n_steps: int, property: str):
    from RiemannMol.atom import optimize
    model = get_model()
    history = optimize(seed, property=property, n_steps=n_steps, model=model)
    return history


def draw_molecules_grid(smiles_list: list[str]) -> bytes | None:
    from rdkit.Chem.Draw import rdMolDraw2D
    mols = []
    labels = []
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            labels.append(smi)
    if not mols:
        return None
    n = len(mols)
    cols = min(n, 2)
    draw_options = rdMolDraw2D.MolDrawOptions()
    draw_options.singleColourBonds = True
    draw_options.bondLineWidth = 2.5
    draw_options.symbolColour = (0, 0, 0)
    draw_options.useBWAtomPalette()
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(400, 350),
        legends=labels,
        useSVG=False,
        drawOptions=draw_options,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


@bot.tree.command(name="gen", description="Generate molecule analogs from a seed SMILES")
@app_commands.describe(
    seed="Seed SMILES string (leave empty for random)",
    n_samples="Number of molecules to generate (1-20, default 5)",
    std="Noise std: 0.3=similar, 1.0=default, 1.5=diverse",
)
@app_commands.rename(n_samples="count")
async def gen(
    interaction: discord.Interaction,
    seed: str | None = None,
    n_samples: int | None = None,
    std: float | None = None,
):
    n_samples = max(1, min(20, n_samples or 5))
    std = max(0.1, min(2.0, std or 1.0))
    actual_seed = seed or random.choice(RANDOM_SEEDS)
    if not seed:
        std = max(std, 1.2)

    await interaction.response.defer(thinking=True)

    try:
        seen = set()
        results = []
        attempts = 0
        while len(results) < n_samples and attempts < 6:
            batch = await asyncio.to_thread(run_generate, actual_seed, n_samples, std)
            for r in batch:
                smi = r.get("smiles", "")
                if smi and smi not in seen:
                    seen.add(smi)
                    results.append(r)
                    if len(results) >= n_samples:
                        break
            attempts += 1
    except Exception as e:
        await interaction.followup.send(f"Error: {e}")
        return

    if not results:
        await interaction.followup.send("No molecules generated. Try a different seed.")
        return

    lines = []
    for i, r in enumerate(results, 1):
        smi = r.get("smiles", "?")
        lines.append(f"`{i}.` **{smi}**")

    desc = "\n".join(lines)
    embed = discord.Embed(
        title="Molecule Generation",
        description=desc,
        color=0x00B4D8,
    )
    embed.set_footer(text=f"Seed: {actual_seed} | std={std} | n={len(results)}")

    if len(results) < 10:
        smi_list = [r.get("smiles", "") for r in results]
        img_bytes = await asyncio.to_thread(draw_molecules_grid, smi_list)
        if img_bytes:
            file = discord.File(io.BytesIO(img_bytes), filename="molecules.png")
            embed.set_image(url="attachment://molecules.png")
            await interaction.followup.send(embed=embed, file=file)
            return

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="optimize", description="Property-guided molecule generation (CMA-ES)")
@app_commands.describe(
    seed="Seed SMILES string",
    property="Property to optimize: qed or logp",
    steps="Number of optimization steps (5-50, default 20)",
)
async def optimize_cmd(
    interaction: discord.Interaction,
    seed: str,
    property: str | None = None,
    steps: int | None = None,
):
    property = property or "qed"
    steps = max(5, min(50, steps or 20))

    await interaction.response.defer(thinking=True)

    try:
        history = await asyncio.to_thread(run_optimize, seed, steps, property)
    except Exception as e:
        await interaction.followup.send(f"Error: {e}")
        return

    if not history:
        await interaction.followup.send("Optimization produced no results.")
        return

    best = history[-1]
    lines = []
    for h in history[-5:]:
        gen = h.get("generation", "?")
        score = h.get("score", "?")
        smi = h.get("smiles", "?")
        lines.append(f"`gen {gen}` score={score:.4f}  **{smi}**")

    embed = discord.Embed(
        title=f"Optimization ({property})",
        description="\n".join(lines),
        color=0x06D6A0,
    )
    embed.add_field(name="Best", value=f"**{best.get('smiles', '?')}**\nscore: {best.get('score', '?')}", inline=False)
    embed.set_footer(text=f"Seed: {seed} | steps={steps}")
    await interaction.followup.send(embed=embed)


if __name__ == "__main__":
    bot.run(TOKEN)
