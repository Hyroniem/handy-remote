# handy-remote: Handy met externe Whisper-server

**handy-remote** is een fork van [cjpais/Handy](https://github.com/cjpais/Handy) en voegt één optie toe:
**Modellen → Transcriptieserver → Externe server gebruiken**. Handy stuurt opnames
dan naar een OpenAI-compatibele server (`{URL}/audio/transcriptions`) en laadt zelf
geen model op de GPU.

## Downloaden

Elke push naar `main` bouwt via **Actions → Fork Build**:

- `handy-fork-ubuntu-22.04-x86_64-unknown-linux-gnu`: `.deb` voor Linux
- `handy-fork-x86_64-pc-windows-msvc`: `.exe`-installer en `.msi` voor Windows
- `handy-fork-aarch64-apple-darwin`: `.dmg` voor Macs met Apple Silicon (M1 en nieuwer)
- `handy-fork-x86_64-apple-darwin`: `.dmg` voor Macs met een Intel-processor

De builds zijn niet ondertekend. Windows SmartScreen waarschuwt daarom bij de eerste
start: kies _Meer informatie → Toch uitvoeren_.

Op de Mac is de app niet door Apple genotariseerd, dus macOS blokkeert de eerste
start. Sleep Handy uit de `.dmg` naar Programma's, probeer hem te openen en kies
daarna in _Systeeminstellingen → Privacy en beveiliging_ onderaan _Toch openen_. Of
in Terminal: `xattr -dr com.apple.quarantine /Applications/Handy.app`. Omdat elke
build een eigen handtekening heeft, moet je na een update de toegang tot
Toegankelijkheid en Microfoon mogelijk opnieuw geven.

## Automatisch bijhouden van upstream

De workflow **Upstream Sync** kijkt elke dag (en op verzoek via _Actions → Upstream Sync → Run workflow_) of cjpais/Handy een nieuwe release `vX.Y.Z` heeft. Zo ja:

1. Hij merget die release in een branch `upstream-sync/vX.Y.Z` en opent een pull request naar `main`.
2. Hij start **Fork Build** op die branch, zodat je ziet of alles compileert.
3. Is de build groen, dan merge je de pull request. Daarna bouwt **Fork Build** op `main` de installers.

Er wordt niets vanzelf in `main` gezet: een schone merge kan toch niet bouwen als upstream code verwijdert die de externe-serveroptie gebruikt. Bij een merge-conflict opent de workflow een issue.

Optioneel: een secret `SYNC_TOKEN` (PAT met rechten `repo` en `workflow`) laat ook de checks `test` en `code quality` op de pull request draaien, en is nodig als upstream iets onder `.github/workflows` wijzigt.

## Een upstream-update handmatig binnenhalen

1. Open https://github.com/Hyroniem/handy-remote. Staat er "This branch is N commits behind
   cjpais/Handy:main", klik dan op **Sync fork → Update branch**.
2. Lukt dat zonder conflict, dan start **Fork Build** vanzelf. Download en installeer
   daarna het nieuwe pakket. Upstream verhoogt het versienummer, dus het pakket
   installeert gewoon over de vorige versie heen.
3. Meldt GitHub een conflict, kies dan **nooit "Discard commits"**, want daarmee
   verdwijnt de externe-serveroptie. Los het lokaal op (zie hieronder) of vraag Claude
   om het te doen.

Een update binnenhalen is nooit verplicht. De fork blijft werken zoals hij is.

## Conflicten oplossen (lokaal)

```bash
cd ~/apps/handy-fork
git fetch upstream
git merge upstream/main   # conflicten oplossen, daarna:
git push origin main
```

Wat de fork ten opzichte van upstream verandert, voor wie de conflicten oplost:

| Bestand                                                                                                     | Wijziging                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src-tauri/src/managers/transcription.rs`                                                                   | `transcribe_remote()` + `encode_wav()`; `transcribe()` en `initiate_model_load()` gaan eerst naar de server als `remote_transcription_enabled` aan staat; bij een fout valt `transcribe()` terug op het lokale model (`remote_transcription_fallback`); `probe_remote_server()` test bij de start van de opname of de server bereikbaar is en met een kort stil testverzoek of hij binnen `remote_transcription_busy_timeout_ms` antwoordt (bezet); zo niet, dan laadt het alvast het lokale model en gebruikt `transcribe()` dat; na een geslaagd serververzoek wordt dat model weer ontladen |
| `src-tauri/src/settings.rs`                                                                                 | velden `remote_transcription_enabled` / `_url` / `_api_key` / `_fallback` / `_busy_timeout_ms`; type `SecretString`; `update_checks_forced_disabled()` geeft altijd `true`                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `src-tauri/src/shortcut/mod.rs`, `lib.rs`                                                                   | vijf `change_remote_transcription_*`-commando's                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `src-tauri/src/actions.rs`                                                                                  | geen live streaming bij een externe server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `src-tauri/src/commands/models.rs`                                                                          | model niet laden bij een modelwissel als de server aan staat                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `src-tauri/Cargo.toml`                                                                                      | reqwest-feature `multipart`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `src-tauri/tauri.conf.json`                                                                                 | `createUpdaterArtifacts: false`, Windows-`signCommand` verwijderd (geen sleutels in de fork)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `src/components/settings/RemoteTranscription.tsx`, `GeneralSettings.tsx`, `settingsStore.ts`, `bindings.ts` | instellingen-UI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `src/i18n/locales/*/translation.json`                                                                       | sleutels `settings.remoteTranscription.*`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `.github/workflows/`                                                                                        | `fork-build.yml` toegevoegd; `main-build.yml` en `nix-check.yml` alleen handmatig; `build.yml` heeft een extra stap "Build with Tauri (unsigned macOS)" zonder Apple-variabelen                                                                                                                                                                                                                                                                                                                                                                                                                |

Controleer na een merge dat `bun run format:check` en `cargo test` slagen, of laat de
workflows _test_ en _code quality_ dat doen.
