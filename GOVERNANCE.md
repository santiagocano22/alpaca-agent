```markdown
# Governance del proyecto

Este documento define el proceso de trabajo. Léelo al inicio de cada sesión, junto con PROMPT.md y PROJECT_STATE.md.

## Modo de trabajo

Trabajas en modo autónomo continuo. Implementas módulos uno tras otro sin pedir aprobación entre ellos. Tomas todas las decisiones de implementación interna. Solo interrumpes al usuario en los TRES casos definidos al final de este documento.

## Reglas innegociables

- Tests pasan al 100% antes de avanzar al siguiente módulo.
-`-W error::DeprecationWarning` activo en pyproject.toml. Cero warnings.
- Wall time por archivo de test < 2s. Usa sleeper inyectado y mocks; no esperas tiempo real.
- Migraciones Alembic: `up + down + up` funciona contra una DB limpia.
- Logging con loguru, nunca `print()`.
- Timestamps tz-aware siempre, UTC en DB.
- Tests pre-existentes en rojo: investigar con `git` para saber si es regresión o test mal escrito. Nunca marcar como "no relacionado".
- Tests que cuelgan: investigar el bug raíz. Nunca bajar parámetros para que pase.
- Warnings de dependencias: silenciar con `filterwarnings` específico o pre-importación controlada en conftest raíz. No usar stubs en `sys.modules` salvo justificación documentada.
- Si tocas la API pública de un módulo aprobado, lo documentas en PROJECT_STATE.md.

## Decisiones que tomas solo (NO preguntar)

- Nombres de variables, funciones, métodos privados, clases internas.
- Orden de validators dentro de un modelo pydantic.
- Estructura de fixtures de tests.
- Tests adicionales por encima del mínimo definido.
- Refactors internos que no cambian la API pública.
- Logging adicional, comentarios, docstrings.
- Elección entre dos librerías equivalentes para algo trivial.
- Cómo silenciar warnings de dependencias externas (siempre que sea limpio).
- Qué scripts de diagnóstico correr para verificar comportamiento de un SDK.
- Cómo estructurar el código dentro de un módulo (archivos auxiliares, helpers internos).

## Decisiones que NUNCA tomas solo

- Cambiar a live trading (`LIVE_TRADING_CONFIRMED=true`).
- Borrar tests pre-existentes en rojo.
- Saltar un módulo del plan.
- Modificar el spec original (PROMPT.md).
- Cambiar la API pública de un módulo ya aprobado sin documentarlo.

## PROJECT_STATE.md — lo mantienes tú

Tú lo creas, tú lo actualizas. El usuario solo lo lee si quiere curiosear. Lo actualizas al terminar cada módulo, antes de mandar el mensaje de cierre al usuario.

Estructura:
```

## Estado del proyecto

### Módulos

* [X] / [ ] por cada uno

### Módulo actual

<nombre>

### Última actualización

<timestamp UTC>

### Tests totales

`<N>` passed

### Acciones pendientes del usuario

<lista o "ninguna">

### Log de módulos completados

#### Módulo X — `<nombre>`

* Tests: `<N>` passed
* Decisiones notables: <1-3 líneas>
* Discrepancias resueltas: `<si las hubo>`
* Pendientes: `<si los hay>`

```

## Los TRES casos en que interrumpes al usuario

### Caso 1: Acción del usuario necesaria

Algo que solo el usuario puede hacer: crear cuenta externa, conseguir token, configurar `.env` con secretos, probar interactivamente algo (mandar mensaje al bot), autorizar paso a live trading.

Formato:
> "Necesito que [acción concreta]. Cuando esté listo, dime y sigo."

### Caso 2: Decisión que requiere juicio del usuario

Decisiones de producto/dinero/UX, no de implementación interna. Ejemplos: comportamiento del risk_manager ante caso ambiguo con dinero real, texto exacto de alertas críticas, política ante eventos no previstos.

Formato:
> "Pregunta: <X>. Opciones: A) <...>, B) <...>. Recomiendo <A o B> porque <razón>. ¿Cuál usamos?"

### Caso 3: Bloqueo real

Algo está roto y no tienes solución obvia después de intentar al menos dos enfoques. Test crítico falla por razón que no entiendes. SDK se comporta distinto a lo documentado y rompe el spec. Conflicto fundamental entre requisitos.

Formato:
> "Bloqueo en <X>. Intenté <Y> y <Z>, ambos fallan porque <...>. Necesito decisión sobre <...>."

## Lo que NO es razón para interrumpir

- Mostrar firmas, fixtures, modelos antes de codear.
- Pedir aprobación de tests adicionales sobre el mínimo.
- Confirmar decisiones internas que esta governance ya autoriza.
- Avisar de progreso intermedio dentro de un módulo.
- Preguntar si seguir con el siguiente módulo cuando no hay acciones pendientes.

## Comunicación al terminar un módulo

Al terminar un módulo, actualizas PROJECT_STATE.md y mandas UN solo mensaje con esta plantilla:
```

Módulo `<X>` terminado.

* Tests: `<N>` passed (total acumulado: `<M>`).
* Wall time: `<T>`s.
* Decisiones notables: <1-3 líneas si las hay, si no, omite>.
* Discrepancias resueltas: <si hubo, si no, omite>.
* Acciones pendientes del usuario: <si las hay, si no, "ninguna">.
  Sigo con módulo `<Y>`.

```

Si "Acciones pendientes del usuario" es "ninguna", arrancas el siguiente módulo inmediatamente sin esperar respuesta.

Si hay acciones pendientes, paras y esperas.

## Recuperación de contexto

Si una sesión empieza sin contexto, lees en orden:
1. PROMPT.md (la spec original).
2. GOVERNANCE.md (este archivo).
3. PROJECT_STATE.md (qué está hecho y qué toca).
4. Si el módulo a hacer es crítico (risk_manager, engine), revisa los reportes de los módulos previos relevantes.
```
